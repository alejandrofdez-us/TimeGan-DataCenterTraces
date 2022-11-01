import itertools

def generate_experiment_command (module, hidden_dim, num_layer, iteration):
    return f'python main_timegan.py --data_name machine_usage_grouped_days_3-4-5-6 --seq_len 8640 --batch_size 100 --metric_iteration 10 --n_samples 150 --internal_discriminative_iteration 80 --internal_predictive_iteration 100' \
           f'--module {module} --iteration {iteration} --num_layer {num_layer} --hidden_dim {hidden_dim}'

def generate_evaluation_metrics_command (module, hidden_dim, num_layer, iteration):
    return f'python evaluation_metrics.py --ori_data_filename data/machine_usage/machine_usage_grouped_days_3-4-5-6.csv --generated_data experiments/iter-{iteration}_num_layer-{num_layer}_hidden_dim-{hidden_dim}_module-{module}/generated_data/ --metrics mmd,dtw,kl,cc,cp,hi,mae,r2,ev,tsne,pca,evolution_figures --trace alibaba2018'


iterations =[500,1000,1500]
num_layers = [2,3,4]
hidden_dims = [4,8,16]
modules = ['gru', 'lstm']

parameterization = [modules, hidden_dims, num_layers, iterations]

parameterization_combinations = list(itertools.product(*parameterization))

commands = [generate_evaluation_metrics_command(module, hidden_dim, num_layer, iteration) for module, hidden_dim, num_layer, iteration in parameterization_combinations]
print (commands)