"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np

def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold=numpy.inf)
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  min= np.min(data,0)
  max = np.max(data, 0)
  numerator = data - min
  denominator = max - min
  norm_data = numerator / (denominator + 1e-7)
  return norm_data, min, max

def MinMaxUnscaler(data, min, max):
  denominator = max - min
  un_norm_data = data * (denominator + 1e-7)
  return un_norm_data + min

def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()
  print("Sine generation")
  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """
  assert data_name in ['stock','energy','trivial','natural','batchtaskday3','alibaba100','alibaba500','alibaba1000','alibaba10k','alibaba50k','alibaba50kcut','alibaba1M', 'alibabacompleto', 'alibabacompletocut', 'alibabacompletocutordered', 'alibabacompletogrouped','alibabacompletogroupedhour']
  print("Cargando datos: ", data_name)
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'trivial':
    ori_data = np.loadtxt('data/mytrivialdata.csv', delimiter=",", skiprows=1)
  elif data_name == 'natural':
    ori_data = np.loadtxt('data/natural_numbers_data.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibaba100':
    ori_data = np.loadtxt('data/day3_first_100.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibaba500':
    ori_data = np.loadtxt('data/day3_first_500.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibaba1000':
    ori_data = np.loadtxt('data/day3_first_1000.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibaba10k':
    ori_data = np.loadtxt('data/day3_first_10000.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibaba50k':
    ori_data = np.loadtxt('data/day3_first_50000.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibaba50kcut':
    ori_data = np.loadtxt('data/day3_first_50000_cut.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibaba1M':
    ori_data = np.loadtxt('data/day3_first_1M.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibabacompleto':
    ori_data = np.loadtxt('data/mu_day3.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibabacompletocut':
    ori_data = np.loadtxt('data/mu_day3_cut.csv', delimiter=",")
  elif data_name == 'alibabacompletocutordered':
    ori_data = np.loadtxt('data/mu_day3_cut_timestamp_first_ordered.csv', delimiter=",")
  elif data_name == 'alibabacompletogrouped':
    ori_data = np.loadtxt('data/mu_day3_grouped.csv', delimiter=",", skiprows=1)
  elif data_name == 'alibabacompletogroupedhour':
    ori_data = np.loadtxt('data/mu_day3_grouped_hours.csv', delimiter=",", skiprows=1)
  elif data_name == 'batchtaskday3':
    ori_data = np.loadtxt('data/batch_task_day3_preprocessed.csv', delimiter=",", skiprows=1)


  #print("Empieza flip")
  #fullprint(ori_data)
  # Flip the data to make chronological data
  # ori_data = ori_data[::-1]
  # print("Datos después de flip:")
  #fullprint(ori_data)
  # Normalize the data
  # ori_data, min, max = MinMaxScaler(ori_data)

  # Preprocess the dataset
  temp_data = []
  print("Cortando los datos.")
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len + 1):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
    #print("corte #:", i)
    #fullprint(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))
  #print("índex idx:")
  #fullprint (idx)
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])

  return data