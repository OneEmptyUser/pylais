import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # THIS IS TO REMOVE A WARNING
import sys
# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pylais')))
import tensorflow as tf
import tensorflow_probability as tfp
from utils import flatTensor3D, repeatTensor3D, general_cov

# class TestUtils:
    
def test_flatTensor3D():
    tmp = tf.constant([[[1, 2, 3], [3, 2, 1]], 
                        [[4, 5, 6], [6, 5, 4]],
                        [[7, 8, 9], [9, 8, 7]]])
    
    expected = tf.constant([[1,2,3],
                            [3,2,1],
                            [4,5,6],
                            [6,5,4],
                            [7,8,9],
                            [9,8,7]])
    assert tf.reduce_all(flatTensor3D(tmp) == expected)
    
def test_repeatTensor3D():
    tmp = tf.constant([[[1, 2, 3], [3, 2, 1]], 
                        [[4, 5, 6], [6, 5, 4]],
                        [[7, 8, 9], [9, 8, 7]]])
    expected = tf.constant([[[1, 2, 3], [1, 2, 3], [3, 2, 1],  [3, 2, 1]], 
                        [[4, 5, 6], [4, 5, 6], [6, 5, 4], [6, 5, 4]],
                        [[7, 8, 9], [7, 8, 9], [9, 8, 7],  [9, 8, 7]]])
    
    assert tf.reduce_all(repeatTensor3D(tmp, 2) == expected)
    
def test_general_cov_full():
    
    cov = tf.constant([[1, 1.2], [1.2, 2]], dtype=tf.float64)
    mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=[1, 1], covariance_matrix=cov)
    tf.random.set_seed(1)
    sample = mvn.sample()
    
    new_state_fn = general_cov(cov)
    tf.random.set_seed(1)
    new_state = new_state_fn(tf.zeros((1,2), dtype=tf.float64)+1, 1)
    assert tf.reduce_all(new_state == sample)
        
def test_general_cov_diag():
    
    cov = tf.constant([1, 2], dtype=tf.float64)
    mvn = tfp.distributions.MultivariateNormalDiag(scale_diag=cov)
    tf.random.set_seed(1)
    sample = mvn.sample()
    
    new_state_fn = general_cov(cov)
    tf.random.set_seed(1)
    new_state = new_state_fn(tf.zeros((1,2), dtype=tf.float64), 1)
    assert tf.reduce_all(new_state == sample)


def test_general_cov_diag1():
    
    cov = tf.constant([1, 2], dtype=tf.float64)
    mvn = tfp.distributions.MultivariateNormalDiag(scale_diag=cov)
    tf.random.set_seed(1)
    sample = mvn.sample()
    
    cov=tf.expand_dims(cov, 0)
    new_state_fn = general_cov(cov)
    tf.random.set_seed(1)
    new_state = new_state_fn(tf.zeros((1,2), dtype=tf.float64), 1)
    assert tf.reduce_all(new_state == sample)
    
def test_general_cov_diag_list():
    cov = tf.constant([1, 2], dtype=tf.float64)
    mvn = tfp.distributions.MultivariateNormalDiag(scale_diag=cov)
    tf.random.set_seed(1)
    sample = mvn.sample()
    
    cov = [1, 2]
    new_state_fn = general_cov(cov)
    tf.random.set_seed(1)
    new_state = new_state_fn(tf.zeros((1,2), dtype=tf.float64), 1)
    assert tf.reduce_all(new_state == sample)

# test_general_cov_diag_list()