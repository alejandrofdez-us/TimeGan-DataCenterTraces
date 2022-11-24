import itertools
import os
import stat
from datetime import datetime


def generate_experiment_command(module, hidden_dim, num_layer, iteration):
    experiment_save_dir = '~/experiments/timegan/alibaba2018/ecms-experiments-5-columns-326-2022-18'
    return f'python main_timegan.py --data_name alibaba2018 --seq_len 288 ' \
           f'--batch_size 100 --metric_iteration 10 --n_samples 10 --internal_discriminative_iteration 80 ' \
           f'--internal_predictive_iteration 100 --experiment_save_dir {experiment_save_dir} ' \
           f'--module {module} --iteration {iteration} --num_layer {num_layer} --hidden_dim {hidden_dim}\n'


iterations = [4500, 5000, 5500, 6000]
num_layers = [3]
hidden_dims = [4]
modules = ['gru']

parameterization = [modules, hidden_dims, num_layers, iterations]

parameterization_combinations = list(itertools.product(*parameterization))

commands = [generate_experiment_command(module, hidden_dim, num_layer, iteration) for
            module, hidden_dim, num_layer, iteration in parameterization_combinations]

sh_filename = "ecms-experiments-" + datetime.now().strftime("%j-%Y-%H-%M") + ".sh"
sh_file = open(sh_filename, "w")
sh_file.writelines(commands)
st = os.stat(sh_filename)
os.chmod(sh_filename, st.st_mode | stat.S_IEXEC)
