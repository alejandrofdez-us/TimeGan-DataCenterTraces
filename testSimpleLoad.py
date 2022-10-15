import os

import numpy as np
import tensorflow as tf

from data_loading import real_data_loading, MinMaxScaler
from utils import random_generator_alt, extract_time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

print(tf.__version__)


def load_tf1(model_directory, Z_mb, ori_data, ori_time, n_samples):
  print('Loading from', model_directory)
  with tf.Graph().as_default() as g:
    with tf.Session() as sess:
      meta_graph = tf.saved_model.load(sess, ["serve"], model_directory)
      sig_def = meta_graph.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
      myinput_x = sig_def.inputs['myinput_x'].name
      myinput_z = sig_def.inputs['myinput_z'].name
      myinput_t = sig_def.inputs['myinput_t'].name


      x_hat = sig_def.outputs['x_hat'].name

      print('Generando datos desde el modelo cargado')
      generated_data_curr = sess.run(x_hat, feed_dict={myinput_z: Z_mb, myinput_x: ori_data, myinput_t: ori_time[:n_samples]})
      print('Se ha terminado la generación')
      generated_data = list()

      for i in range(n_samples):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)
      # Renormalization
      generated_data = generated_data * max_val
      generated_data = generated_data + min_val

      generated_data_np_array = np.asarray(generated_data)
      i = 0
      directory_name = model_directory+"/generated_data/"
      os.makedirs(directory_name, exist_ok=True)
      for generated_sample in generated_data_np_array:
        np.savetxt(directory_name + "sample-" + str(i) + ".csv", generated_sample, delimiter=",", fmt='%f')
        i = i + 1



# random_generator variables
n_samples = 100
dim = 4
z_dim = dim
data_name = "alibabacompletocut"
seq_len = 100
model_directory = "experiments/alibabacompletocut10-13-2022-21-22/models"


ori_data = real_data_loading(data_name, seq_len)
ori_time, max_seq_len = extract_time(ori_data)

ori_data, min_val, max_val = MinMaxScaler(ori_data)

Z_mb = random_generator_alt(n_samples, z_dim, ori_time, max_seq_len)

load_tf1(model_directory, Z_mb, ori_data, ori_time, n_samples)
