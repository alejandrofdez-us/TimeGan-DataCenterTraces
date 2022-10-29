import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
for i in range(0, 32):

    day_name = "day_"+f'{i:02}'
    print("computando día: ", day_name)
    directory_name = 'data/trazas_google/'
    ori_data = np.loadtxt(directory_name+'instance_usage_5min_sample_'+day_name+'.csv', delimiter=",", skiprows=1)
    seq_len = len(ori_data[:, 0])

    column_names = ['cpu_usage', 'mem_usage', 'assigned_mem', 'cycles_per_instruction']
    for column_name in column_names:
        index = column_names.index(column_name)
        column_data = ori_data[:, index]
        axis = [0, seq_len, 0, 1]
        if column_name == 'cycles_per_instruction':
            axis = [0, seq_len, 0, np.amax(column_data)]

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
