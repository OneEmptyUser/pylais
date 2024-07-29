import tensorflow as tf
# from tensorflow_probability.distributions import MultivariateNormalFullCovariance as mvn
import tensorflow_probability as tfp
mvn = tfp.distributions.MultivariateNormalFullCovariance

def spatial(means, samples, cov):
    """
    Calculate the spatial denominator for each sample.
    
    Calculate the spatial denominator as the average of the density evaluation
    on each sample of all the proposals that were adapted at the same time
    that the proposal that originated the sample.

    Parameters
    ----------
        means: tensorflow.Tensor
            Tensor of shape (N, T, dim) representing the means.
        samples: tensorflow.Tensor
            Tensor of shape (_, n_samples, _) representing the samples.
        cov_down: tensorflow.Tensor
            The covariance matrix of the proposals

    Returns
    -------
        dens: tensorflow.Tensor
            Tensor of shape (N, n_samples, 1) representing the spatial denominator.
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
    """
    Calculate the temporal denominator for each sample.
    
    Calculate the temporal denominator as the average of the density evaluation
    on each sample of all the proposals that were adapted in the same chain as
    the proposal that originated the sample.

    Parameters
    ----------
        means: tensorflow.Tensor
            Tensor of shape (N, T, dim) representing the means.
        samples: tensorflow.Tensor
            Tensor of shape (_, n_samples, _) representing the samples.
        cov: tensorflow.Tensor
            The covariance matrix of the proposals

    Returns
    -------
        dens: tensorflow.Tensor
            Tensor of shape (N, n_samples, 1) representing the temporal denominator
    """
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
    """
    Calculate the total denominator for each sample.
    
    Calculate the total denominator as the average of the density evaluation
    of all the proposals on each sample.
    
    Parameters:
        flatted_means: tensorflow.Tensor
            The tensor of means with shape (N, T, dim).
        flatted_samples: tensorflow.Tensor
            The tensor of samples with shape (_, n_samples, _).
        cov: tensorflow.Tensor
            The covariance matrix of the proposals with shape (dim, dim).

    Returns:
        dens: tensorflow.Tensor
            The tensor of denominators with shape (N, n_samples, 1).
    """
    aux_fn = tf.function(
        lambda x: tf.math.reduce_mean(mvn(loc=x, covariance_matrix=cov).prob(flatted_means))
        )
    
    dens = tf.map_fn(
        fn=aux_fn,
        elems=flatted_samples
    )
    return dens