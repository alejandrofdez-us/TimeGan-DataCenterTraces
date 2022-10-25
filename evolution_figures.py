import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle




def create_figure(ori_column_values_array, generated_column_values,axis, name, path_to_save_metrics):
    f, ax = plt.subplots(1)
    plt.rcParams["figure.figsize"] = (18, 3)
    i=1
    cycol = cycle('grcmk')

    for ori_column_values in ori_column_values_array:
        plt.plot(ori_column_values, c=next(cycol), label="Original_"+str(i), linewidth=1)
        i += 1

    plt.plot(generated_column_values, c="blue", label="Synthetic", linewidth=2)
    if axis:
        plt.axis(axis)
    else:
        plt.xlim([0,8640])
    plt.title('PCA plot')
    plt.xlabel('time')
    plt.ylabel(name)
    ax.legend()
    plt.savefig(path_to_save_metrics + name + '.png')
    plt.close()


def create_usage_evolution(generated_data_sample, ori_data, ori_data_sample, path_to_save_metrics):
    seq_len = len(ori_data_sample[:,0])

    cpu_usage_column = 0
    generate_figures_by_column(cpu_usage_column, "cpu",generated_data_sample,ori_data, ori_data_sample, path_to_save_metrics, seq_len)

    mem_usage_column = 1
    generate_figures_by_column(mem_usage_column, "mem", generated_data_sample, ori_data, ori_data_sample, path_to_save_metrics, seq_len)

    return


def generate_figures_by_column(column_number, column_name, generated_data_sample, ori_data, ori_data_sample, path_to_save_metrics, seq_len):
    create_figure(ori_column_values_array=[ori_data_sample[:, column_number]], generated_column_values=generated_data_sample[:, column_number], axis=[0, seq_len, 0, 100], name=column_name+'_usage', path_to_save_metrics=path_to_save_metrics)
    generate_figures_grouped_by_minutes_various_ori_samples(1/6, column_number, column_name, generated_data_sample, ori_data, path_to_save_metrics, seq_len, 5)
    generate_figures_grouped_by_minutes_various_ori_samples(1, column_number, column_name, generated_data_sample, ori_data, path_to_save_metrics, seq_len, 5)
    generate_figures_grouped_by_minutes_various_ori_samples(10, column_number, column_name, generated_data_sample, ori_data, path_to_save_metrics, seq_len, 5)
    generate_figures_grouped_by_minutes_various_ori_samples(30, column_number, column_name, generated_data_sample, ori_data, path_to_save_metrics, seq_len, 5)
    generate_figures_grouped_by_minutes_various_ori_samples(60, column_number, column_name, generated_data_sample, ori_data, path_to_save_metrics, seq_len, 5)

def generate_figures_grouped_by_minutes_various_ori_samples (minutes, column_number, column_name, generated_data_sample, ori_data, path_to_save_metrics, seq_len, n_ori_samples=1):

    delta_ori_column_array = [compute_grouped_delta_from_sample(column_number, minutes, get_ori_data_sample(seq_len, ori_data), seq_len) for i in
                        range(n_ori_samples)]

    delta_gen_column = compute_grouped_delta_from_sample(column_number, minutes, generated_data_sample, seq_len)

    max_y_value = max(np.amax(delta_ori_column_array), np.amax(delta_gen_column))
    min_y_value = min(np.amin(delta_ori_column_array), np.amin(delta_gen_column))
    create_figure(ori_column_values_array=delta_ori_column_array, generated_column_values=delta_gen_column, axis=[0,seq_len//(minutes * 6),min_y_value,max_y_value],
                      name=column_name + '_grouped_usage_delta_'+str(round(minutes, 2))+'min', path_to_save_metrics=path_to_save_metrics)


def compute_grouped_delta_from_sample(column_number, minutes, data_sample, seq_len):
    sample_column = data_sample[:, column_number]
    sample_column_splitted = np.array_split(sample_column, seq_len // (minutes * 6))
    sample_column_mean = [np.mean(batch) for batch in sample_column_splitted]
    delta_sample_column = -np.diff(sample_column_mean)
    return delta_sample_column


def get_ori_data_sample(seq_len, ori_data):
    ori_data_sample_start = random.randrange(0, len(ori_data) - seq_len)
    ori_data_sample_end = ori_data_sample_start + seq_len
    ori_data_sample = ori_data[ori_data_sample_start:ori_data_sample_end]
    return ori_data_sample