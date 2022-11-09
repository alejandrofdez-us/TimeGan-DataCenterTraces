# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

import numpy as np
import math


def KLDivergenceUnivariate(array_1, array_2, num_bins=100):
  min_array1 = array_1.min()
  min_array2 = array_2.min()
  min_all = min(min_array1, min_array2)
  max_array1 = array_1.max()
  max_array2 = array_2.max()
  max_all = max(max_array1, max_array2)
  range_values = (min_all, max_all)
  p = np.histogram(array_1, bins=np.linspace(range_values[0], range_values[1], num_bins + 1))[0] / len(array_1)
  q = np.histogram(array_2, bins=np.linspace(range_values[0], range_values[1], num_bins + 1))[0] / len(array_2)
  KL_p_m = sum([p[i] * np.log(p[i] / q[i]) if (p[i] != 0 and q[i] != 0) else 0 for i in range(len(p))])
  KL_q_m = sum([q[i] * np.log(q[i] / p[i]) if (p[i] != 0 and q[i] != 0) else 0 for i in range(len(p))])
  return KL_p_m, KL_q_m

def JSDistance(array_1, array_2, num_bins=100):
    KL_p_m, KL_q_m = KLDivergenceUnivariate(array_1, array_2, num_bins=num_bins)
    JS_p_q = math.sqrt((KL_p_m + KL_q_m) / 2)
    return JS_p_q

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.

  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.

  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).

  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]


  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))