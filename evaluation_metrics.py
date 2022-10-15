import argparse
import os
import random
import statistics

import numpy as np
import scipy

from metrics.kl import KLdivergence
from metrics.mmd import mmd_rbf
from dtaidistance import dtw_ndim
from dtaidistance import dtw
import torch
import pandas
from metrics.visualization_metrics import visualization


def preprocess_dataset(ori_data, seq_len):
    temp_data = []
    print("Cortando los datos.")
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


def main (args):
    directory_name = args.generated_data_dir + "/metrics/"
    os.makedirs(directory_name, exist_ok=True)
    metrics_list = [metric for metric in args.metrics.split(',')]


    ori_data = np.loadtxt(args.ori_data_filename, delimiter=",", skiprows=1)
    ori_data[:, [1, 0]] = ori_data[:, [0, 1]] # timestamp como primera columna
    ori_data_histogram = torch.tensor(np.histogram(ori_data)[-1])
    if "tsne" in metrics_list or "pca" in metrics_list:
        generate_visualization_figures(args, directory_name, metrics_list, ori_data)

    metrics_results = {}
    avg_results = {}
    for metric in metrics_list:
        print ('Computando: ', metric)
        metrics_results[metric] = []
        dtw_series = []
        for filename in os.listdir(args.generated_data_dir):
            ori_data_sample_start = random.randrange(0, len(ori_data)-args.ori_sample_size)
            ori_data_sample_end = ori_data_sample_start+args.ori_sample_size
            ori_data_sample = ori_data[ori_data_sample_start:ori_data_sample_end]
            f = os.path.join(args.generated_data_dir, filename)
            if os.path.isfile(f): # checking if it is a file
                generated_data_sample = np.loadtxt(f, delimiter=",")
                generated_data_sample[:, [1, 0]] = generated_data_sample[:, [0, 1]] #timestamp como primera columna
                if metric == 'mmd': #mayor valor más distintas son
                    mmd_result = mmd_rbf(X=ori_data_sample, Y=generated_data_sample)
                    metrics_results[metric].append(mmd_result)
                if metric == 'dtw': #mayor valor más distintas son
                    res = dtw_ndim.distance(generated_data_sample, ori_data_sample)
                    metrics_results[metric].append(res)
                if metric == 'kl':
                    dist = KLdivergence(ori_data, generated_data_sample)
                    metrics_results[metric].append(dist)
                if metric == 'cc':
                    ori_data_sample_numpy_pearson = np.corrcoef(ori_data_sample[:generated_data_sample.shape[0]])
                    generated_data_sample_numpy_pearson = np.corrcoef(generated_data_sample)
                    covariance_diff_matrix = ori_data_sample_numpy_pearson - generated_data_sample_numpy_pearson
                    l1_norms_avg = np.mean([np.linalg.norm(row) for row in covariance_diff_matrix])
                    metrics_results[metric].append(l1_norms_avg)


    for metric, results in metrics_results.items():
        if metric == 'kl' or metric == 'dtw' or metric == 'mmd' or metric == 'cc':
            avg_results[metric] = statistics.mean(metrics_results[metric])


    with open(directory_name + 'metrics.txt', 'w') as f:
        f.write(repr(avg_results))
        f.write("\n")
        f.write(repr(metrics_results))


def generate_visualization_figures(args, directory_name, metrics_list, ori_data):
    ori_data_for_visualization = preprocess_dataset(ori_data, args.seq_len)
    generated_data = []
    n_samples = 0
    for filename in os.listdir(args.generated_data_dir):
        f = os.path.join(args.generated_data_dir, filename)
        if os.path.isfile(f):  # checking if it is a file
            n_samples = n_samples + 1
            generated_data_sample = np.loadtxt(f, delimiter=",")
            generated_data_sample[:, [1, 0]] = generated_data_sample[:, [0, 1]]  # timestamp como primera columna
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
        '--generated_data_dir',
        type=str)
    parser.add_argument(
        '--metrics',
        default='mmd',
        type=str)
    parser.add_argument(
        '--ori_sample_size',
        default='1000',
        type=int)
    parser.add_argument(
        '--seq_len',
        type=int)

    args = parser.parse_args()

    main(args)



