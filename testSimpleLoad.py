import os

import numpy as np
import tensorflow as tf

from data_loading import real_data_loading, MinMaxScaler
from utils import random_generator_alt, extract_time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

print(tf.__version__)


def load_tf1(path, Z_mb, ori_data, ori_time, n_samples):
  print('Loading from', path)
  with tf.Graph().as_default() as g:
    with tf.Session() as sess:
      meta_graph = tf.saved_model.load(sess, ["serve"], path)
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
      directory_name = "models/generated_data/"
      os.makedirs(directory_name, exist_ok=True)
      for generated_sample in generated_data_np_array:
        np.savetxt(directory_name + "sample-" + str(i) + ".csv", generated_sample, delimiter=",", fmt='%f')
        i = i + 1



# random_generator variables
n_samples = 150
dim = 4
z_dim = dim
data_name = "alibaba50kcut"
seq_len = 100
model_filename = "models"


ori_data = real_data_loading(data_name, seq_len)
ori_time, max_seq_len = extract_time(ori_data)

ori_data, min_val, max_val = MinMaxScaler(ori_data)

Z_mb = random_generator_alt(n_samples, z_dim, ori_time, max_seq_len)

load_tf1(model_filename, Z_mb, ori_data, ori_time, n_samples)


#
# def load_pb(path_to_pb):
#     with tf.gfile.GFile(path_to_pb, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph
#
# model_filename = "models/saved_model.pb"
#
# restored_graph = load_pb(model_filename)
# graph = tf.Graph()
# with restored_graph.as_default():
#     with tf.Session() as sess:
#         tf.saved_model.loader.load(
#             sess,
#             [tag_constants.SERVING],
#             'models',
#         )
#
#         myinput_x = graph.get_tensor_by_name('myinput_x:0')
#         myinput_z = graph.get_tensor_by_name('myinput_z:0')
#         myinput_t = graph.get_tensor_by_name('myinput_t:0')
#         prediction = restored_graph.get_tensor_by_name('prediction:0') #probar quizás a poner el nombre largo ese "recovery_1/fully_connected/Sigmoid:0")
#
#         #random_generator variables
#         n_samples = 10
#         dim = 4
#         z_dim = dim
#         data_name = "alibaba50kcut"
#         seq_len = 100
#
#         ori_data = real_data_loading(data_name, seq_len)
#         ori_time, max_seq_len = extract_time(ori_data)
#
#         ori_data, min_val, max_val = MinMaxScaler(ori_data)
#
#         Z_mb = random_generator_alt(n_samples, z_dim, ori_time, max_seq_len)
#
#         generated_data_curr = sess.run(prediction, feed_dict={
#             myinput_z: Z_mb,
#             myinput_x: ori_data,
#             myinput_t: ori_time[:n_samples]
#         })
#
#         print(generated_data_curr)
#
#
