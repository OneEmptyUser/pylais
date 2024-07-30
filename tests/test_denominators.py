import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # THIS IS TO REMOVE A WARNING
import sys
# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pylais')))
import tensorflow as tf
import tensorflow_probability as tfp
from denominators import all_, spatial, temporal
from utils import flatTensor3D, repeatTensor3D

mvn = tfp.distributions.MultivariateNormalFullCovariance(
    covariance_matrix=tf.eye(2, dtype=tf.float64)
)
tf.random.set_seed(27)


N = 3
n_iter = 10
n_per_sample = 2
dim = 2

fake_means = []
for i in range(3):
  matrix = [tf.zeros(2, dtype=tf.float64)]
  for t in range(n_iter-1):
    matrix.append(mvn.sample() + matrix[-1])
  tensor = tf.stack(matrix)
  tensor = tf.expand_dims(tensor, axis=0)
  fake_means.append(tensor)
fake_means = tf.concat(fake_means, axis=0)

repeated_means = repeatTensor3D(fake_means, n_per_sample)
flatted_repeated_means = flatTensor3D(repeated_means)
flatted_means = flatTensor3D(fake_means)
mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(2, dtype=tf.float64),
                                                            covariance_matrix=tf.eye(2, dtype=tf.float64))

flatted_samples = mvn.sample(n_per_sample*n_iter*N) + flatted_repeated_means
samples = tf.reshape(flatted_samples, (N,n_iter*n_per_sample, dim))


def test_all_():
    
    expected_weights = []
    for n in range(flatted_samples.shape[0]):
        mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=flatted_samples[n],
                                                                 covariance_matrix=tf.eye(2, dtype=tf.float64))
        expected_weights.append(tf.math.reduce_mean(mvn.prob(flatted_means)).numpy())
    
    assert tf.reduce_all(tf.constant(expected_weights) == all_(flatted_means, flatted_samples, tf.eye(2, dtype=tf.float64)))
    
def test_temporal():
    expected_weights = []
    for n in range(samples.shape[0]):
        # weights_chain = []
        for t in range(samples.shape[1]):
            loc = samples[n, t, :]
            mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=loc,
                                                                     covariance_matrix=tf.eye(2, dtype=tf.float64))
            expected_weights.append(tf.math.reduce_mean(mvn.prob(fake_means[n, :, :])).numpy())
        # weights_chain = tf.stack(weights_chain)
        # expected_weights.append(weights_chain.numpy())
    
    assert tf.reduce_all(tf.constant(expected_weights) == temporal(fake_means, samples, tf.eye(2, dtype=tf.float64)))
        

def test_spatial():
    expected_weights = []
    
    n_per_sample = samples.shape[1]//fake_means.shape[1]
    for n in range(samples.shape[0]):
        for t in range(samples.shape[1]):
            loc = samples[n, t, :]
            mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=loc,
                                                                     covariance_matrix=tf.eye(2, dtype=tf.float64))
            expected_weights.append(tf.math.reduce_mean(mvn.prob(repeated_means[:, t, :])).numpy())
        # weights_chain = tf.stack(weights_chain)
        # expected_weights.append(weights_chain.numpy())
    
    assert tf.reduce_all(tf.constant(expected_weights) == spatial(fake_means, samples, tf.eye(2, dtype=tf.float64)))
    
    
    
    
    
    
    
# test_all_()