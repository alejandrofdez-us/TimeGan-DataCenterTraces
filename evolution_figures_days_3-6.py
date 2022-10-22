import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

ori_data = np.loadtxt('data/machine_usage/machine_usage_grouped_days_3-4-5-6.csv', delimiter=",", skiprows=0)

seq_len = len(ori_data[:, 0])
axis = [0, seq_len, 0, 100]

f, ax = plt.subplots(1)
plt.rcParams["figure.figsize"] = (18, 3)
plt.plot(ori_data[:, 0], c="blue", label="Ori_data", linewidth=1)
plt.axis(axis)
plt.title('PCA plot')
plt.xlabel('time')
plt.ylabel('cpu_usage')
ax.legend()
plt.savefig('cpu_usage_days_3-6.png')
plt.close()

f, ax = plt.subplots(1)
plt.rcParams["figure.figsize"] = (18, 3)
plt.plot(ori_data[:, 1], c="blue", label="Original data", linewidth=1)
plt.axis(axis)
plt.title('PCA plot')
plt.xlabel('time')
plt.ylabel('mem_usage')
ax.legend()
plt.savefig('mem_usage_days_3-6.png')
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
