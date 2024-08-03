import pytest
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # THIS IS TO REMOVE A WARNING
import sys
# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pylais')))
import tensorflow as tf
from lais import Lais
# ------------------------------------------------------------------------------

def target(theta):
    return tf.exp(-0.5*tf.linalg.norm(theta)**2)

def create_targets():
    target1 = lambda theta: tf.exp(-0.5*tf.linalg.norm(theta - 2)**2)
    target2 = lambda theta: tf.exp(-0.5*tf.linalg.norm(theta - 3)**2)
    target3 = lambda theta: tf.exp(-0.5*tf.linalg.norm(theta + 1)**2)
    return [target1, target2, target3]

def test_init():
    errors = []
    theta  = tf.constant((1, 1), dtype=tf.float64)
    try:
        myLais = Lais(target)
        myLais.loglikelihood(theta)
    except:
        errors.append("Error in __init__")
    assert not errors
    
def test_upper():
    N, T, dim = 3, 10, 2
    initial_points = tf.constant([(-5, -5), (5, 5), (2, 2)], dtype=tf.float64)
    mcmc_settings = {
        "method": "mcmc",
        "cov" : tf.eye(dim, dtype=tf.float64),
        }
    myLais = Lais(target)
    
    errors = []
    try:
        means = myLais.upper_layer(T, N, initial_points,mcmc_settings["method"], mcmc_settings)
    except Exception as e:
        errors.append(f"Error in upper_layer:\n{e}")
    
    try:
        assert means.samples.shape == (N, T, dim)
    except Exception as e:
        errors.append(f"Error with the shape of means:\n{e}")
    assert not errors
    
def test_upper_targets():
    N, T, dim = 3, 10, 2
    initial_points = tf.constant([(-5, -5), (5, 5), (2, 2)], dtype=tf.float64)
    mcmc_settings = {
        "method": "mcmc",
        "cov" : tf.eye(dim, dtype=tf.float64),
        }
    targets = create_targets()
    errors = []
    try:
        myLais = Lais(target)
        means = myLais.upper_layer(T, N, initial_points,mcmc_settings["method"],
                                   mcmc_settings, targets=targets)
        if means.samples.shape != (N, T, dim):
            raise(Exception("Error with the shape of means"))
    except Exception as e:
        errors.append(f"Error in upper_layer:\n{e}")
        pytest.fail(f"Error in upper_layer:\n{e}")
    assert not errors
    
def test_upper_hmc():
    N, T, dim = 3, 10, 2
    initial_points = tf.constant([(-5, -5), (5, 5), (2, 2)], dtype=tf.float64)
    mcmc_settings = {
        "method": "hmc",
        "step_size": 0.01,
        "num_leapfrog_steps": 10,
        }
    targets = create_targets()
    errors = []
    try:
        myLais = Lais(target)
        means = myLais.upper_layer(T, N, initial_points,mcmc_settings["method"],
                                   mcmc_settings, targets=targets)
        if means.samples.shape != (N, T, dim):
            raise(Exception("Error with the shape of means"))
    except Exception as e:
        errors.append(f"Error in upper_layer:\n{e}")
        pytest.fail(f"Error in upper_layer:\n{e}")
    assert not errors
    
def test_upper_mala():
    N, T, dim = 3, 10, 2
    initial_points = tf.constant([(-5, -5), (5, 5), (2, 2)], dtype=tf.float64)
    mcmc_settings = {
        "method": "mala",
        "step_size": 0.01
        }
    targets = create_targets()
    errors = []
    try:
        myLais = Lais(target)
        means = myLais.upper_layer(T, N, initial_points,mcmc_settings["method"],
                                   mcmc_settings, targets=targets)
        if means.samples.shape != (N, T, dim):
            raise(Exception("Error with the shape of means"))
    except Exception as e:
        errors.append(f"Error in upper_layer:\n{e}")
        pytest.fail(f"Error in upper_layer:\n{e}")
    assert not errors
    
def test_upper_nuts():
    N, T, dim = 3, 10, 2
    initial_points = tf.constant([(-5, -5), (5, 5), (2, 2)], dtype=tf.float64)
    mcmc_settings = {
        "method": "nuts",
        "step_size": 0.01
        }
    targets = create_targets()
    errors = []
    try:
        myLais = Lais(target)
        means = myLais.upper_layer(T, N, initial_points,mcmc_settings["method"],
                                   mcmc_settings, targets=targets)
        if means.samples.shape != (N, T, dim):
            raise(Exception("Error with the shape of means"))
    except Exception as e:
        errors.append(f"Error in upper_layer:\n{e}")
        pytest.fail(f"Error in upper_layer:\n{e}")
    assert not errors
    
def test_upper_slice():
    N, T, dim = 3, 10, 2
    initial_points = tf.constant([(-5, -5), (5, 5), (2, 2)], dtype=tf.float64)
    mcmc_settings = {
        "method": "slice",
        "step_size": 0.01,
        "max_doublings": 10
        }
    targets = create_targets()
    errors = []
    try:
        myLais = Lais(target)
        means = myLais.upper_layer(T, N, initial_points,mcmc_settings["method"],
                                   mcmc_settings, targets=targets)
        if means.samples.shape != (N, T, dim):
            raise(Exception("Error with the shape of means"))
    except Exception as e:
        errors.append(f"Error in upper_layer:\n{e}")
        pytest.fail(f"Error in upper_layer:\n{e}")
    assert not errors