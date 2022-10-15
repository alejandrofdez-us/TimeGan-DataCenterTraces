import os

import numpy as np

directory_name = 'experiments/random/generated0-1000/'
os.makedirs(directory_name, exist_ok=True)

for i in range(100):
    random_sample = np.random.randint(0,1000,(100, 4))
    np.savetxt(directory_name + "sample_" + str(i) + ".csv", random_sample, delimiter=",", fmt='%f')


random_sample = np.random.randint(0,1000,(10000, 4))
np.savetxt(directory_name + "sample_" + str(10000) + ".csv", random_sample, delimiter=",", fmt='%f')
