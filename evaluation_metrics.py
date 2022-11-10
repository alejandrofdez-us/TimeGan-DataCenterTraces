import argparse
import os
import random
import re
import statistics

import numpy as np
import pandas as pd
import scipy

from argparse import Namespace

from evolution_figures import create_usage_evolution
from metrics.kl import KLdivergence, JSDistance
from metrics.kl import KLDivergenceUnivariate
from metrics.mmd import mmd_rbf
from dtaidistance import dtw_ndim
from dtaidistance import dtw
import torch
import pandas
from metrics.visualization_metrics import visualization

import sklearn.metrics as metrics


def main (args):
    if (args.recursive == 'true'):
        root_dir = args.experiment_dir
        first_level_dirs =[]
        for subdir, dirs, files in os.walk(root_dir):
            first_level_dirs = dirs
            break
        is_header_printed = False
        for dir in first_level_dirs:
            args.experiment_dir = root_dir+dir
            try:
                print("Computing metrics for directory ", dir)
                saved_metrics, metrics_values, saved_experiment_parameters = compute_metrics(args)
                parameters_keys, parameters_values = extract_experiment_parameters(saved_experiment_parameters)
                if(not is_header_printed):
                    with open(root_dir + 'experiments_metrics.csv', 'w') as f:
                        print("Printing header")
                        f.write('experiment_dir_name;' + parameters_keys + saved_metrics + '\n')
                    is_header_printed = True

                with open(root_dir + 'experiments_metrics.csv', 'a') as f:
                    f.write(dir+';'+parameters_values+metrics_values+ '\n')

            except Exception as e:
                print('Error computing experiment dir:', args.experiment_dir)

        print ("\nCSVs for all experiments metrics results saved in:\n", root_dir + 'experiments_metrics.csv')
    else:
        compute_metrics(args)


def extract_experiment_parameters(saved_experiment_parameters):
    saved_experiment_parameters_dict = dict(
        item.split("=") for item in saved_experiment_parameters.replace('Namespace(', '').replace(')', '').split(", "))
    parameters_values = ''
    parameters_keys = ''
    for parameter_value in saved_experiment_parameters_dict.values():
        parameters_values += parameter_value + ';'
    for parameter_key in saved_experiment_parameters_dict.keys():
        parameters_keys += parameter_key + ';'
    return parameters_keys, parameters_values


def compute_metrics (args):

    metrics_list, path_to_save_metrics, saved_experiments_parameters, saved_metrics, dataset_info = initialization(args)

    ori_data = np.loadtxt(args.ori_data_filename, delimiter=",", skiprows=0)
    #ori_data[:, [1, 0]] = ori_data[:, [0, 1]] # timestamp como primera columna
    if "tsne" in metrics_list or "pca" in metrics_list:
        generate_visualization_figures(args, path_to_save_metrics, metrics_list, ori_data)
        metrics_list.remove("tsne")
        metrics_list.remove("pca")

    metrics_results = {}
    avg_results = {}
    for metric in metrics_list:
        print ('Computando: ', metric)
        metrics_results[metric] = []
        if (metric == 'mmd' or metric == 'dtw' or metric == 'kl' or metric == 'hi' or metric == 'ks'):
            for column in range(ori_data.shape[1]):
                metrics_results[metric + '-' + str(column)] = []
                if(metric == 'kl'):
                    metrics_results[metric + '-JSD-' + str(column)] = []

        n_files_iteration = 0
        for filename in os.listdir(args.experiment_dir+'/generated_data'):
            ori_data_sample = get_ori_data_sample(args, ori_data)
            f = os.path.join(args.experiment_dir+'/generated_data', filename)
            if os.path.isfile(f): # checking if it is a file
                generated_data_sample = np.loadtxt(f, delimiter=",")
                computed_metric = 0
                if metric == 'mmd': #mayor valor más distintas son
                    computed_metric = mmd_rbf(X=ori_data_sample, Y=generated_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric+'-'+str(column)].append(mmd_rbf(generated_data_sample[:,column].reshape(-1, 1), ori_data_sample[:,column].reshape(-1, 1)))
                if metric == 'dtw': #mayor valor más distintas son
                    computed_metric = compute_dtw(generated_data_sample, ori_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric+'-'+str(column)].append(compute_dtw(generated_data_sample[:,column].reshape(-1, 1), ori_data_sample[:,column].reshape(-1, 1)))
                if metric == 'kl': #mayor valor peor
                    computed_metric = KLdivergence(ori_data, generated_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(KLDivergenceUnivariate(ori_data_sample[:,column].reshape(-1, 1), generated_data_sample[:,column].reshape(-1, 1))[0])
                        metrics_results[metric + '-JSD-' + str(column)].append(JSDistance(ori_data_sample[:, column].reshape(-1, 1), generated_data_sample[:, column].reshape(-1, 1)))
                if metric == 'ks':  # menor valor mejor
                    computed_metric = compute_ks(generated_data_sample, ori_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(compute_ks(generated_data_sample[:,column].reshape(-1, 1), ori_data_sample[:,column].reshape(-1, 1)))
                if metric == 'cc': #mayor valor peor. covarianza
                    computed_metric = compute_cc(generated_data_sample, ori_data_sample)
                if metric == 'cp': #mayor valor peor. coeficiente de pearson
                    computed_metric = compute_cp(generated_data_sample, ori_data_sample)
                if metric == 'hi':  # mayor valor peor
                    computed_metric = compute_hi(generated_data_sample, ori_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric+'-'+str(column)].append(compute_hi(generated_data_sample[:,column].reshape(-1, 1), ori_data_sample[:,column].reshape(-1, 1)))
                if metric == 'evolution_figures':
                    if n_files_iteration % 10 == 0: #generates a 10% of the figures
                        create_usage_evolution(generated_data_sample, ori_data, ori_data_sample, path_to_save_metrics+'figures/', n_files_iteration, dataset_info)
                if metric != 'evolution_figures':
                    metrics_results[metric].append(computed_metric)

                n_files_iteration += 1

    for metric, results in metrics_results.items():
        if metric != 'tsne' and metric != 'pca' and metric != 'evolution_figures':
            avg_results[metric] = statistics.mean(metrics_results[metric])
    saved_metrics, metrics_values, = save_metrics(avg_results, metrics_results, path_to_save_metrics, saved_experiments_parameters, saved_metrics)
    return saved_metrics, metrics_values, saved_experiments_parameters


def initialization(args):
    path_to_save_metrics = args.experiment_dir + "/evaluation_metrics/"
    f = open(args.experiment_dir + '/parameters.txt', 'r')
    saved_experiments_parameters = f.readline()
    f = open(args.experiment_dir + '/metrics.txt', 'r')
    saved_metrics = f.readline()
    args.seq_len = int(re.search("\Wseq_len=([^,}]+)\)", saved_experiments_parameters).group(1))
    os.makedirs(path_to_save_metrics, exist_ok=True)
    os.makedirs(path_to_save_metrics+'/figures/', exist_ok=True)

    metrics_list = [metric for metric in args.metrics.split(',')]

    if (args.trace == 'alibaba2018'):
        dataset_info = {
            "timestamp_frequency_secs": 10,
            "column_config": {
                "cpu": {
                    "column_index": 0,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "mem": {
                    "column_index": 1,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "net_in": {
                    "column_index": 2,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "net_out": {
                    "column_index": 3,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                }
            }
        }
    elif (args.trace == 'google2019'):
        dataset_info = {
            "timestamp_frequency_secs": 300,
            "column_config": {
                "cpu": {
                    "column_index": 0,
                    "y_axis_min": 0,
                    "y_axis_max": 1
                },
                "mem": {
                    "column_index": 1,
                    "y_axis_min": 0,
                    "y_axis_max": 1
                },
                "assigned_mem": {
                    "column_index": 2,
                    "y_axis_min": 0,
                    "y_axis_max": 1
                },
                "cycles_per_instruction": {
                    "column_index": 3
                }
            }
        }
    elif (args.trace == 'azure_v2'):
        dataset_info = {
            "timestamp_frequency_secs": 300,
            "column_config": {
                "cpu_total": {
                    "column_index": 0
                },
                "mem_total": {
                    "column_index": 1
                }
            }
        }
    elif (args.trace == 'reddit'):
        dataset_info = {
            "timestamp_frequency_secs": 3600,
            "column_config": {
                "interactions": {
                    "column_index": 0
                }
            }
        }

    return metrics_list, path_to_save_metrics, saved_experiments_parameters, saved_metrics, dataset_info


def preprocess_dataset(ori_data, seq_len):
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len + 1):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


def results_for_excel(avg_results):
    metrics_values = ''
    computed_metrics = ''
    for metric_name in avg_results:
        computed_metrics += metric_name+';'
        metrics_values += str(avg_results[metric_name]).replace('.', ',')+';'

    return computed_metrics, metrics_values

def save_metrics(avg_results, metrics_results, path_to_save_metrics, saved_experiments_parameters, saved_metrics):
    data_name = re.search("\Wdata_name=([^,}]+)", saved_experiments_parameters).group(1).replace("'","")
    iterations = re.search("\Witeration=([^,}]+)", saved_experiments_parameters).group(1)
    seq_len = re.search("\Wseq_len=([^,}]+)\)", saved_experiments_parameters).group(1)
    with open(path_to_save_metrics + '/metrics-'+data_name+'-iterations-'+iterations+'-seq_len'+seq_len+'.txt', 'w') as f:
        f.write(saved_experiments_parameters + '\n\n')
        f.write(saved_metrics +'\n\n')
        f.write(repr(avg_results) + '\n')
        computed_metrics, metrics_values = results_for_excel(avg_results)
        f.write( 'Results of the following metrics: ' + computed_metrics + ' in spanish locale Excel format:' + '\n' + metrics_values + '\n')
        f.write(repr(metrics_results))
    print("Metrics saved in file", f.name)
    return computed_metrics, metrics_values

def compute_ks (generated_data_sample, ori_data_sample):
    column_indexes = range(generated_data_sample.shape[1])
    return statistics.mean([scipy.stats.ks_2samp(generated_data_sample[:,column_index], ori_data_sample[:,column_index])[0] for column_index in column_indexes])

def compute_dtw(generated_data_sample, ori_data_sample):
    sample_lenght = len(generated_data_sample)
    processed_generated_data_sample = np.insert(generated_data_sample, 0, np.ones(sample_lenght, dtype=int), axis=1)
    processed_generated_data_sample = np.insert(processed_generated_data_sample, 0, range(sample_lenght), axis=1)
    processed_ori_data_sample = np.insert(ori_data_sample, 0, np.ones(sample_lenght, dtype=int), axis=1)
    processed_ori_data_sample = np.insert(processed_ori_data_sample, 0, range(sample_lenght), axis=1)

    return dtw_ndim.distance_fast(processed_generated_data_sample, processed_ori_data_sample)

def compute_cp(generated_data_sample, ori_data_sample):
    #normalized_ori_data_sample = normalize_start_time_to_zero(ori_data_sample)
    #normalized_generated_data_sample = normalize_start_time_to_zero(generated_data_sample)
    ori_data_sample_pearson = np.corrcoef(ori_data_sample)
    generated_data_sample_pearson = np.corrcoef(generated_data_sample)
    correlation_diff_matrix = ori_data_sample_pearson - generated_data_sample_pearson
    l1_norms_avg = np.mean([np.linalg.norm(row) for row in correlation_diff_matrix])
    return l1_norms_avg

def compute_cc(generated_data_sample, ori_data_sample):
    #normalized_ori_data_sample = normalize_start_time_to_zero(ori_data_sample)
    #normalized_generated_data_sample = normalize_start_time_to_zero(generated_data_sample)
    ori_data_sample_covariance = np.cov(ori_data_sample)
    generated_data_covariance = np.cov(generated_data_sample)
    covariance_diff_matrix = ori_data_sample_covariance - generated_data_covariance
    l1_norms_avg = np.mean([np.linalg.norm(row) for row in covariance_diff_matrix])
    return l1_norms_avg

def compute_hi(generated_data_sample, ori_data):
    #normalized_ori_data_sample = normalize_start_time_to_zero(ori_data)
    #normalized_generated_data_sample = normalize_start_time_to_zero(generated_data_sample)
    histogram_diff_matrix=[]
    for column in range(0,ori_data.shape[1]):
       ori_data_column_values = ori_data[:,column]
       ori_histogram,ori_bin_edges = np.histogram(ori_data_column_values)
       generated_data_column_values = generated_data_sample[:, column]
       generated_histogram, generated_bin_edges = np.histogram(generated_data_column_values)
       column_histogram_diff = ori_histogram - generated_histogram
       histogram_diff_matrix.append(column_histogram_diff)
    histogram_diff_matrix = np.asmatrix(histogram_diff_matrix)
    l1_norms_histogram_diff = np.apply_along_axis(np.linalg.norm, 1, histogram_diff_matrix)

    l1_norms_histogram_diff_avg = l1_norms_histogram_diff.mean()

    return l1_norms_histogram_diff_avg

def normalize_start_time_to_zero (sample):
    timestamp_column =  sample[:,0]
    min_timestamp = np.min(timestamp_column)
    normalized_timestamp_column=timestamp_column - min_timestamp
    sample[:,0]=normalized_timestamp_column
    return sample

def get_ori_data_sample(args, ori_data):
    ori_data_sample_start = random.randrange(0, len(ori_data) - args.seq_len)
    ori_data_sample_end = ori_data_sample_start + args.seq_len
    ori_data_sample = ori_data[ori_data_sample_start:ori_data_sample_end]
    return ori_data_sample


def generate_visualization_figures(args, directory_name, metrics_list, ori_data):
    ori_data_for_visualization = preprocess_dataset(ori_data, args.seq_len)
    generated_data = []
    n_samples = 0
    for filename in os.listdir(args.experiment_dir+'/generated_data'):
        f = os.path.join(args.experiment_dir+'/generated_data', filename)
        if os.path.isfile(f):  # checking if it is a file
            n_samples = n_samples + 1
            generated_data_sample = np.loadtxt(f, delimiter=",")
            generated_data.append(generated_data_sample)
    if "tsne" in metrics_list:
        visualization(ori_data=ori_data_for_visualization, generated_data=generated_data, analysis='tsne',
                      n_samples=n_samples, path_for_saving_images=directory_name)
    if "pca" in metrics_list:
        visualization(ori_data=ori_data_for_visualization, generated_data=generated_data, analysis='pca',
                      n_samples=n_samples, path_for_saving_images=directory_name)


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ori_data_filename',
        default='data/mu_day3_cut.csv',
        type=str)
    parser.add_argument(
        '--experiment_dir',
        type=str)
    parser.add_argument(
        '--metrics',
        default='mmd',
        type=str)
    parser.add_argument(
        '--seq_len',
        type=int)
    #implementar diccionario de configuración por tipo de traza
    parser.add_argument(
        '--trace',
        default='alibaba2018',
        type=str)
    parser.add_argument(
        '--recursive',
        default='false',
        type=str)

    args = parser.parse_args()
    main(args)