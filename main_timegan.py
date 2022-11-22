"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from datetime import datetime

import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def main(args, experiment_root_directory_name):
    """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
    - n_samples: number of samples to be genereated
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
    ## Data loading
    if args.data_name in ['stock', 'energy', 'google_instance_week1', 'alibaba2018']:
        ori_data = real_data_loading(args.data_name, args.seq_len)
        print(args.data_name + ' dataset is ready.')
    elif args.data_name == 'sine':
        # Set number of samples and its dimensions
        no, dim = 10000, 5
        ori_data = sine_data_generation(no, args.seq_len, dim)
        print(args.data_name + ' dataset is ready.')

    ## Synthetic data generation by TimeGAN
    # Set newtork parameters
    parameters = dict()
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations'] = args.iteration
    parameters['batch_size'] = args.batch_size
    parameters['n_samples'] = args.n_samples
    n_samples = args.n_samples

    generated_data = timegan(ori_data, parameters, experiment_root_directory_name)
    print('Finish Synthetic Data Generation')

    ## Performance metrics
    # Output initialization
    metric_results = dict()
    metric_results['discriminative'] = 0
    metric_results['predictive'] = 0

    # 1. Discriminative Score
    # discriminative_score = list()
    # i=1
    # for _ in range(args.metric_iteration):
    #   print("Iteracion", i, "de", args.metric_iteration,"de discriminative score")
    #   i += 1
    #   temp_disc = discriminative_score_metrics(ori_data, generated_data, args.internal_discriminative_iteration)
    #   discriminative_score.append(temp_disc)
    #
    # metric_results['discriminative'] = np.mean(discriminative_score)

    # # 2. Predictive score
    # predictive_score = list()
    # i=1
    # for tt in range(args.metric_iteration):
    #   print("Iteracion",i, "de", args.metric_iteration,"de predictive score")
    #   i+=1
    #   temp_pred = predictive_score_metrics(ori_data, generated_data, args.internal_predictive_iteration)
    #   predictive_score.append(temp_pred)
    #
    # print("Finalizando scores de prediccion")
    # metric_results['predictive'] = np.mean(predictive_score)
    # # Print discriminative and predictive scores
    # print(metric_results)

    # 3. Visualization (PCA and tSNE)
    print("Creando graficas PCA y tSNE")
    visualization(ori_data, generated_data, 'pca', experiment_root_directory_name, n_samples)
    visualization(ori_data, generated_data, 'tsne', experiment_root_directory_name, n_samples)

    return ori_data, generated_data, metric_results


if __name__ == '__main__':

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['sine', 'stock', 'energy', 'trivial', 'natural', 'google_instance_week1', 'alibaba2018'],
        default='stock',
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=24,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru', 'lstm', 'lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--experiment_save_dir',
        default='experiments/',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=24,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=50000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=128,
        type=int)
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=10,
        type=int)
    parser.add_argument(
        '--n_samples',
        help='number of samples to be generated',
        default=150,
        type=int)
    parser.add_argument(
        '--internal_discriminative_iteration',
        help='internal iterations of discriminative scoring',
        default=2000,
        type=int)
    parser.add_argument(
        '--internal_predictive_iteration',
        help='internal iterations of predictive scoring',
        default=5000,
        type=int)

    args = parser.parse_args()

    # experiment_root_directory_name = "experiments/" + args.data_name + '_' + str(args.iteration) + '-' + datetime.now().strftime("%j-%Y-%H-%M")+"/"
    experiment_root_directory_name = args.experiment_save_dir + "/" + 'iter-' + str(
        args.iteration) + '_' + 'num_layer-' + str(args.num_layer) + '_' + 'hidden_dim-' + str(
        args.hidden_dim) + '_' + 'module-' + str(args.module) + '/'
    # Calls main function
    ori_data, generated_data, metrics = main(args, experiment_root_directory_name)

    print("Metrics")
    print(metrics)

    generated_data_np_array = np.asarray(generated_data)
    i = 0
    directory_name = experiment_root_directory_name + "generated_data/"
    os.makedirs(experiment_root_directory_name, exist_ok=True)
    os.makedirs(directory_name, exist_ok=True)

    with open(experiment_root_directory_name + 'metrics.txt', 'w') as f:
        f.write(repr(metrics))

    text_file = open(experiment_root_directory_name + "parameters.txt", "w")
    n = text_file.write(str(args))
    text_file.close()

    for generated_sample in generated_data_np_array:
        np.savetxt(directory_name + "sample_" + str(i) + ".csv", generated_sample, delimiter=",", fmt='%f')
        i = i + 1
