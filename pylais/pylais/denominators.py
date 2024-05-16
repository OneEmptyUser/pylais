import tensorflow as tf
# from tensorflow_probability.distributions import MultivariateNormalFullCovariance as mvn
import tensorflow_probability as tfp
mvn = tfp.distributions.MultivariateNormalFullCovariance

def spatial(means, samples, cov):
    """
    Calculate the spatial density for the given means, samples, and covariance matrix.

    Parameters:
    - means: numpy array of shape (N, T, dim) representing the means.
    - samples: numpy array of shape (_, n_samples, _) representing the samples.
    - cov_down: covariance matrix.

    Returns:
    - dens: numpy array of shape (N, n_samples, 1) representing the spatial density
    """
    
    N, T, dim = means.shape
    _, n_samples, _ = samples.shape
    dens = []
    k = n_samples // T # number of samples per proposal
    
    for n in range(N):
        for t in range(n_samples):
            dens.append(
                tf.math.reduce_mean(mvn(loc=samples[n, t, :], covariance_matrix=cov).prob(means[:, t//k, :]))
                )
    dens = tf.stack(dens) 
    return dens

def temporal(means, samples, cov):
    N, T, dim = means.shape
    _, n_samples, _ = samples.shape
    
    dens = []
    for n in range(N):
        for t in range(n_samples):
            dens.append(
                tf.math.reduce_mean(mvn(loc=samples[n, t, :], covariance_matrix=cov).prob(means[n, :, :]))
                )
            
    dens = tf.stack(dens)
    return dens

def all_(flatted_means, flatted_samples, cov):
    aux_fn = tf.function(
        lambda x: tf.math.reduce_mean(mvn(loc=x, covariance_matrix=cov).prob(flatted_means))
        )
    
    dens = tf.map_fn(
        fn=aux_fn,
        elems=flatted_samples
    )
    return dens