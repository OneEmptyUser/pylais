import tensorflow as tf
# from tensorflow_probability.distributions import MultivariateNormalFullCovariance as mvn
import tensorflow_probability as tfp
mvn = tfp.distributions.MultivariateNormalTriL

def spatial(means, samples, proposal_settings):
    """
    Calculate the spatial denominator for each sample.

    Calculate the spatial denominator as the average of the density evaluation
    on each sample of all the proposals that were adapted at the same time
    that the proposal that originated the sample.

    Parameters
    ----------
    means : tensorflow.Tensor
        Tensor of shape (N, T, dim) representing the means.
    samples : tensorflow.Tensor
        Tensor of shape (_, n_samples, _) representing the samples.
    proposal_settings : dict
        Dictionary containing the proposal settings, including the covariance matrix and the proposal type.

    Returns
    -------
    dens : tensorflow.Tensor
        Tensor of shape (n_samples,) representing the spatial denominator.
    """
    
    N, T, dim = means.shape
    dType = means.dtype
    _, n_samples, _ = samples.shape
    
    cov = proposal_settings.get("cov", tf.eye(dim), dType)
    scale = tf.linalg.cholesky(cov)
    if proposal_settings.get("proposal_type", "gaussian") == "gaussian":
        proposal = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(dim, dtype=dType),
                                                            scale_tril=scale)
    
    if proposal_settings.get("proposal_type", "gaussian") == "student":
        df = proposal_settings.get("df", dim + 1)
        proposal = tfp.distributions.MultivariateStudentTLinearOperator(df,
                                                                        loc=tf.zeros(dim, dtype=dType),
                                                                        scale=tf.linalg.LinearOperatorLowerTriangular(scale))
        
    dens = []
    k = n_samples // T # number of samples per proposal
    
    for n in range(N):
        for t in range(n_samples):
            loc = samples[n, t, :]
            dens.append(
                tf.math.reduce_mean(proposal.prob(means[:, t//k, :] - loc))
                )
    dens = tf.stack(dens) 
    return dens

def temporal(means, samples, proposal_settings):
    """
    Calculate the temporal denominator for each sample.

    Calculate the temporal denominator as the average of the density evaluation
    on each sample of all the proposals that were adapted at the same time
    that the proposal that originated the sample.

    Parameters
    ----------
    means : tensorflow.Tensor
        Tensor of shape (N, T, dim) representing the means.
    samples : tensorflow.Tensor
        Tensor of shape (_, n_samples, _) representing the samples.
    proposal_settings : dict
        Dictionary containing the proposal settings, including the covariance matrix and the proposal type.

    Returns
    -------
    dens : tensorflow.Tensor
        Tensor of shape (n_samples,) representing the temporal denominator.
    """
    
    N, T, dim = means.shape
    _, n_samples, _ = samples.shape
    dType = means.dtype
    
    cov = proposal_settings.get("cov", tf.eye(dim, dtype=dType))
    scale = tf.linalg.cholesky(cov)
    if proposal_settings.get("proposal_type", "gaussian") == "gaussian":
        proposal = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(dim, dtype=dType),
                                                            scale_tril=scale)
    
    if proposal_settings.get("proposal_type", "gaussian") == "student":
        df = proposal_settings.get("df", dim + 1)
        proposal = tfp.distributions.MultivariateStudentTLinearOperator(df,
                                                                        loc=tf.zeros(dim, dtype=dType),
                                                                        scale=tf.linalg.LinearOperatorLowerTriangular(scale))
    
    dens = []
    for n in range(N):
        for t in range(n_samples):
            loc = samples[n, t, :]
            dens.append(
                tf.math.reduce_mean(proposal.prob(means[n, :, :] - loc))
                )
            
    dens = tf.stack(dens)
    return dens

def all_(flatted_means, flatted_samples, proposal_settings):
    """
    Calculate the total denominator for each sample.

    Calculate the total denominator as the average of the density evaluation
    of all the proposals on each sample.

    Parameters
    ----------
    flatted_means : tensorflow.Tensor
        The tensor of means with shape (N*T, dim).
    flatted_samples : tensorflow.Tensor
        The tensor of samples with shape (n_samples, _).
    proposal_settings : dict
        Dictionary containing the proposal settings, including the covariance matrix and the proposal type.

    Returns
    -------
    dens : tensorflow.Tensor
        The tensor of denominators with shape (n_samples,).
    """
    
    
    _, dim = flatted_means.shape
    dType = flatted_means.dtype
    cov = proposal_settings.get("cov", tf.eye(dim, dtype=dType))
    scale = tf.linalg.cholesky(cov)
    if proposal_settings.get("proposal_type", "gaussian") == "gaussian":
        proposal = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(dim, dtype=dType),
                                                            scale_tril=scale)
    
    if proposal_settings.get("proposal_type", "gaussian") == "student":
        df = proposal_settings.get("df", dim + 1)
        proposal = tfp.distributions.MultivariateStudentTLinearOperator(df,
                                                                        loc=tf.zeros(dim, dtype=dType),
                                                                        scale=tf.linalg.LinearOperatorLowerTriangular(scale))
    
    aux_fn = tf.function(
        lambda x: tf.math.reduce_mean(proposal.prob(flatted_means - x))
        )
    
    dens = tf.map_fn(
        fn=aux_fn,
        elems=flatted_samples
    )
    return dens