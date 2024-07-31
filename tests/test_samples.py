import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # THIS IS TO REMOVE A WARNING
import sys
# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pylais')))
import tensorflow as tf
# import tensorflow_probability as tfp
from samples import ISSamples
import pytest
from matplotlib.testing.decorators import _cleanup_cm
from matplotlib.pyplot import close as plt_close, switch_backend, subplots
switch_backend('Agg')

@pytest.fixture
def samples_instance():
    samples = tf.random.normal((10, 2), dtype=tf.float64)
    weights = tf.ones((10,))
    norm_weights = weights/tf.math.reduce_sum(weights)
    return samples, norm_weights, ISSamples(samples, weights)

def test_Z(samples_instance):
    s, w, obj = samples_instance
    assert obj.Z == 1

def test_len(samples_instance):
    s, w, obj = samples_instance
    assert len(obj) == 10

def test_getitem(samples_instance):
    s, w, obj = samples_instance
    assert tf.reduce_all(s[3] == obj[3][0]) and tf.reduce_all(w[3] == obj[3][1])
    
def test_ess(samples_instance):
    s, w, obj = samples_instance
    assert abs(obj.ess - 10) < 1e-5
    
def test_resample(samples_instance):
    s, w, obj = samples_instance
    tf.random.set_seed(0)
    idx = tf.random.categorical(tf.math.log([w]), 5)
    new_s = tf.gather(s, tf.squeeze(idx))
    new_obj = obj.resample(5, seed=0)
    assert tf.reduce_all(new_s == new_obj.samples)
    
@_cleanup_cm()
def test_scatter(samples_instance):
    s, w, obj = samples_instance
    # Create a scatter plot and ensure it does not raise an exception
    obj.scatter()
    plt_close()
    
@_cleanup_cm()
def test_scatter_with_limits(samples_instance):
    s, w, obj = samples_instance
    # Create a scatter plot with x and y limits and ensure it does not raise an exception
    obj.scatter(xlim=(-1, 1), ylim=(-1, 1))
    plt_close()

@_cleanup_cm()
def test_scatter_with_custom_axis(samples_instance):
    s, w, obj = samples_instance
    # Create a custom axis
    fig, ax = subplots()
    # Create a scatter plot with a custom axis and ensure it does not raise an exception
    obj.scatter(axis=ax)
    plt_close(fig)