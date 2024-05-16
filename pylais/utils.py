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
    def logp(theta):
        return loglikelihood(theta) + logprior(theta) if logprior else loglikelihood(theta)
    return logp

def scatter(tensor, xlim=None, ylim=None, axis=None):
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
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper


def flatTensor3D(tensor):
    N, T, dim = tensor.shape
    return tf_reshape(tensor, (N*T, dim))

def repeatTensor3D(tensor, each):
    return tf_repeat(tensor, each, axis=1)

@tf_function
def run_mcmc(kernel, num_results, num_burnin_steps, current_state):
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
    
    