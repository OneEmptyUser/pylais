import pytest
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # THIS IS TO REMOVE A WARNING
import sys
# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pylais')))
import tensorflow as tf
from samples import mcmcSamples
from matplotlib.testing.decorators import _cleanup_cm
from matplotlib.pyplot import close as plt_close, switch_backend, subplots
switch_backend('Agg')


@pytest.fixture
def sample_instance():
    # Create sample data
    samples = tf.random.normal((3, 100, 2), dtype=tf.float64)  # 3 chains, 100 iterations, 2 parameters
    return mcmcSamples(samples)


def test_initialization(sample_instance):
    assert isinstance(sample_instance, mcmcSamples)
    assert sample_instance.samples.shape == (3, 100, 2)
    
def test_trace(sample_instance):
    try:
        sample_instance.trace()
        sample_instance.trace(chains=[0, 1])
    except Exception as e:
        pytest.fail(f"Trace plotting raised an exception: {e}")

    plt_close()
    
def test_scatter(sample_instance):
    try:
        sample_instance.scatter()
        sample_instance.scatter(xlim=(0, 1), ylim=(0, 1), chains=[0, 1], dims=(0, 1))
    except Exception as e:
        pytest.fail(f"Scatter plotting raised an exception: {e}")

    plt_close()
    
def test_auto_corr_plot(sample_instance):
    try:
        sample_instance.autoCorrPlot()
        sample_instance.autoCorrPlot(component=1, max_lag=20)
    except Exception as e:
        pytest.fail(f"Autocorrelation plotting raised an exception: {e}")

    plt_close()
    
def test_cumulative_mean(sample_instance):
    _, ax = subplots()
    try:
        sample_instance.cumulativeMean()
        sample_instance.cumulativeMean(chains=[1, 2], axis=ax)
    except Exception as e:
        pytest.fail(f"Cumulative mean plotting raised an exception: {e}")

    plt_close() 
    
def test_invalid_chains(sample_instance):
    with pytest.raises(ValueError):
        sample_instance.trace(chains='invalid')  # Should raise a ValueError
        
def test_invalid_dims(sample_instance):
    with pytest.raises(ValueError):
        sample_instance.scatter(dims=(0, 10))  # Assuming there are only 2 dimensions