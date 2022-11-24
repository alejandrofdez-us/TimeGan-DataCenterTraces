import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
for i in range(1, 9):

    day_name = "day_"+str(i)
    print("computando día: ", day_name)
    directory_name = 'data/trazas_google/'
    ori_data = np.loadtxt(directory_name+'instance_usage_'+day_name+'_cut.csv', delimiter=",", skiprows=0)
    seq_len = len(ori_data[:, 0])

    column_names = ['cpu_usage', 'mem_usage', 'assigned_mem', 'cycles_per_instruction']
    for column_name in column_names:
        index = column_names.index(column_name)
        column_data = ori_data[:, index]
        axis = [0, seq_len, np.amin(column_data), np.amax(column_data)]
        plt.rcParams["figure.figsize"] = (18, 3)
        f, ax = plt.subplots(1)
        plt.plot(column_data, c="blue", label="Original data", linewidth=1)
        plt.axis(axis)
        plt.title('AVG '+column_name)
        plt.xlabel('time')
        plt.ylabel(column_name)
        ax.legend()
        plt.savefig(directory_name+column_name+'_'+day_name+'.png')
        plt.close()

#
#
# def create_figure(ori_column_values_array, generated_column_values,axis, name, path_to_save_metrics):
#     f, ax = plt.subplots(1)
#     plt.rcParams["figure.figsize"] = (18, 3)
#     i=1
#     cycol = cycle('grcmk')
#
#     for ori_column_values in ori_column_values_array:
#         plt.plot(ori_column_values, c=next(cycol), label="Original_"+str(i), linewidth=1)
#         i += 1
#
#     plt.plot(generated_column_values, c="blue", label="Synthetic", linewidth=1)
#     if axis:
#         plt.axis(axis)
#     else:
#         plt.xlim([0,8640])
#     plt.title('PCA plot')
#     plt.xlabel('time')
#     plt.ylabel(name)
#     ax.legend()
#     plt.savefig(path_to_save_metrics + name + '.png')
#     plt.close()
