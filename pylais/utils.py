from time import time
from tensorflow import reshape as tf_reshape
from tensorflow import function as tf_function
from tensorflow import constant as tf_constant
from tensorflow import is_tensor as tf_is_tensor
from tensorflow import repeat as tf_repeat
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# def new_state_rw_fn()
def returnKernel(kernel, logposterior, settings={}):
    """Build the transition kernel for the MCMC algorithm.

    Args:
        kernel: str
            The type of algorithm to use.
        logposterior: function
            The log posterior function to use.
        settings: dict (optional)
            A dictionary with the settings of each specific algorithm. Defaults to {}.

    Returns:
        _type_: _description_
    """
    if kernel=="rw":
        cov = settings.get("cov", None)
        if cov is None:
            return tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=logposterior
            )
        return tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=logposterior,
            new_state_fn=general_cov(cov)
            )
    elif kernel=="hmc":
        step_size = settings.get("step_size", 0.01)
        num_leapfrog_steps = settings.get("num_leapfrog_steps", 10)
        return tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=logposterior,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps
            )
    elif kernel=="mala":
        step_size = settings.get("step_size", 0.01)
        return tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=logposterior,
            step_size=step_size
        )
    elif kernel=="nuts":
        step_size = settings.get("step_size", 0.01)
        return tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=logposterior,
            step_size=step_size
            )
    # elif kernel=="replica":
    #     return tfp.mcmc.ReplicaExchangeMC(
    #         target_log_prob_fn=logposterior
    #         )
    elif kernel=="slice":
        step_size = settings.get("step_size", 0.01)
        max_doublings = settings.get("max_doublings",10)
        return tfp.mcmc.SliceSampler(
            target_log_prob_fn=logposterior,
            step_size=step_size,
            max_doublings=max_doublings
        )
    
    return kernel

def general_cov(cov):
    """
    Return a function that generates the new state coming from a Gaussian with
    the given covariance matrix.

    Parameters:
    - cov: (list, tuple, or tf.Tensor) 
        If it is a list or tuple, it is a diagonal covariance matrix; if it is a tf.Tensor,
        it is a covariance matrix.
    
    Returns:
    - new_state_fn: (function)
        A function that generates the new state coming from a Gaussian with
        the given covariance matrix.
    """
    
    # if it is a list of number
    if not tf_is_tensor(cov):
        if isinstance(cov, (list, tuple)):
            # cov = tf.linalg.diag(cov)
            cov = tf.cast(cov, dtype=tf.float64)
            mvn = tfp.distributions.MultivariateNormalDiag(scale_diag=cov)
    else:
        # if it is a matrix with one row or one column
        if (cov.shape[0]==1 and cov.shape[1]>1) or (cov.shape[0]>1 and cov.shape[1]==1):
            # cov = tf.linalg.diag(tf.squeeze(cov))
            cov = tf.cast(cov, dtype=tf.float64)
            mvn = tfp.distributions.MultivariateNormalDiag(scale_diag=cov)
        else:
            mvn = tfp.distributions.MultivariateNormalFullCovariance(covariance_matrix=cov)
    def new_state_fn(state_parts, seed_list):
        next_state_parts = [
            state_part + mvn.sample()
            for state_part in state_parts
        ]    
        return next_state_parts
    return new_state_fn

def buildModelLogp(loglikelihood, logprior=None):
    """
    Build a log posterior function for a model.

    Parameters:
        loglikelihood: function
            A function that calculates the log likelihood of the model.
        logprior: function, (optional)
            A function that calculates the log prior of the model. Defaults to None.

    Returns:
        logp: (function)
            A function that calculates the log posterior of the model.

    """
    def logp(theta):
        return loglikelihood(theta) + logprior(theta) if logprior else loglikelihood(theta)
    return logp

def scatter(tensor, xlim=None, ylim=None, axis=None):
    """
    Plot a scatter plot of a 2D or 3D tensor.

    Parameters:
        tensor: (tf.Tensor)
            The tensor to be plotted. It can be a 2D or 3D tensor.
        xlim: tuple, (optional)
            The x-axis limits for the plot. Defaults to None.
        ylim: tuple, (optional)
            The y-axis limits for the plot. Defaults to None.
        axis: matplotlib.axes._subplots.AxesSubplot, (optional)
            The axis to plot on. Defaults to None.

    Returns:
        None
    """
    
    if axis:
        ax = axis
    else:
        fig, ax = plt.subplots()
    if tensor.ndim == 3:
        for i in range(len(tensor)):
            ax.scatter(tensor[i, :, 0], tensor[i, :, 1], s=2)
    if tensor.ndim == 2:
        ax.scatter(tensor[:, 0], tensor[:, 1], s=2)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.show()
    
def timeit(func):
    """
    Decorator function that measures the execution time of a given function.

    Args:
        func: (callable)
            The function to be timed.

    Returns:
        callable: The wrapped function that measures the execution time.
    """
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper


def flatTensor3D(tensor):
    """
    Reshape a 3D tensor into a 2D tensor by flattening the second dimension.

    Parameters:
        tensor: tensorflow.Tensor
            The input 3D tensor of shape (N, T, dim).

    Returns:
        tf.Tensor: The flattened 2D tensor of shape (N*T, dim).
    """
    N, T, dim = tensor.shape
    return tf_reshape(tensor, (N*T, dim))

def repeatTensor3D(tensor, each):
    """
    Repeat a 3D tensor along the second dimension.

    Parameters:
        tensor: tensorflow.Tensor
            The input 3D tensor of shape (N, T, dim).
        each: int
            The number of times to repeat each element along the second dimension.

    Returns:
        tf.Tensor: The tensor with repeated elements along the second dimension. The shape is (N, T*each, dim).
    """
    return tf_repeat(tensor, each, axis=1)

@tf_function
def run_mcmc(kernel, num_results, num_burnin_steps, current_state):
    """
    Run the MCMC algorithm using the given kernel.

    Parameters:
        kernel: tfp.mcmc.Kernel
            The MCMC kernel to use.
        num_results: int
            The number of samples to generate.
        num_burnin_steps: int
            The number of burn-in steps to perform.
        current_state: tf.Tensor
            The initial state of the MCMC chain.

    Returns:
        tf.Tensor: The generated samples from the MCMC chain.
    """
    if not tf_is_tensor(current_state):
        current_state = tf_constant(current_state)
    samples = tfp.mcmc.sample_chain(
        num_results=num_results,
        kernel=kernel,
        num_burnin_steps=num_burnin_steps,
        current_state = current_state,
        trace_fn=lambda current_state, kernel_results: kernel_results
    )
    return samples
if __name__ == "__main__":
    import tensorflow as tf
    a = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print(a)
    print(tf.reshape(a, (4, 3)))
    
    