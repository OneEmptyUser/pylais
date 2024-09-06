import tensorflow as tf
# from tensorflow_probability.distributions import MultivariateNormalFullCovariance as mvn
import tensorflow_probability as tfp
mvn = tfp.distributions.MultivariateNormalTriL

def spatial(means, samples, proposal_settings):
    """
    Calculate the spatial denominator for each sample.

    Calculate the spatial denominator as the average of the density evaluation
    on each sample of all the proposals that were adapted at the same time
    that the proposal that originated the sample:

    .. math::

        \Phi(x_{n,t}) = \dfrac{1}{N}\sum_{i=1}^N q(x_{n,t} | \mu_{i, t})

    Parameters
    ----------
    means : tensorflow.Tensor, shape (N, T, dim)
        Tensor of means.
    samples : tensorflow.Tensor, shape (N, N*T*M, dim)
        Tensor of samples.
    proposal_settings : dict
        Dictionary containing the proposal settings, including the covariance matrix and the proposal type. The possible
        keys for this dictionary are:
        
        - "cov": the covariance matrix of the proposal distribution.
        - "proposal_type": the type of proposal distribution. Possible values are "gaussian" and "student".
        - "df": the degrees of freedom of the student-t distribution. Only used if "proposal_type" is "student".

    Returns
    -------
    dens : tensorflow.Tensor
        Tensor of shape (n_samples,) representing the spatial denominator.
    """
    
    N, T, dim = means.shape
    dType = means.dtype
    _, n_samples, _ = samples.shape
    
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
    k = n_samples // T # number of samples per proposal
    
    for n in range(N):
        for t in range(n_samples):
            loc = samples[n, t, :]
            dens.append(
                tf.math.reduce_mean(proposal.prob(means[:, t//k, :] - loc))
                )
    dens = tf.stack(dens) 
    return dens

def spatial2(means, flatted_samples, proposal_settings):
    N, T, dim = means.shape
    dType = means.dtype
    n_samples, dim = flatted_samples.shape
    
    M = n_samples // (T * N)
    
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
    
    @tf.function
    def aux_f(i):
        sample = flatted_samples[i]
        false_t = i % (M*T)
        t = false_t // M
        return tf.math.reduce_mean(proposal.prob(means[:, t, :] - sample))
    
    dens = tf.map_fn(
        fn=aux_f, 
        elems=tf.range(n_samples),
        dtype=dType)
    
    return dens
 
def temporal(means, samples, proposal_settings):
    """
    Calculate the temporal denominator for each sample.

    Calculate the temporal denominator as the average of the density evaluation
    on each sample of all the proposals that were adapted at the same time
    that the proposal that originated the sample.
    
    .. math::
    
        \Phi(x_{n,t}) = \dfrac{1}{T}\sum_{k=1}^T q(x_{n,t} | \mu_{n, k})

    Parameters
    ----------
    means : tensorflow.Tensor, shape (N, T, dim)
        Tensor of means.
    samples : tensorflow.Tensor, shape (N, N*T*M, dim)
        Tensor of samples.
    proposal_settings : dict
        Dictionary containing the proposal settings, including the covariance matrix and the proposal type. The possible
        keys for this dictionary are:
        
        - "cov": the covariance matrix of the proposal distribution.
        - "proposal_type": the type of proposal distribution. Possible values are "gaussian" and "student".
        - "df": the degrees of freedom of the student-t distribution. Only used if "proposal_type" is "student".

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

def temporal2(means, flatted_samples, proposal_settings):
    N, T, dim = means.shape
    dType = means.dtype
    n_samples, dim = flatted_samples.shape
    
    M = n_samples // (T * N)
    
    cov = proposal_settings.get("cov", tf.eye(dim, dtype=dType))
    scale = tf.linalg.cholesky(cov)
    if proposal_settings.get("proposal_type", "gaussian") == "gaussian":
        proposal = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(dim, dtype=dType),
                                                            scale_tril=scale)
    
    elif proposal_settings.get("proposal_type", "gaussian") == "student":
        df = proposal_settings.get("df", dim + 1)
        proposal = tfp.distributions.MultivariateStudentTLinearOperator(df,
                                                                        loc=tf.zeros(dim, dtype=dType),
                                                                        scale=tf.linalg.LinearOperatorLowerTriangular(scale))

    def aux_f(i):
        sample = flatted_samples[i]
        n = i // (T * M)
        return tf.math.reduce_mean(proposal.prob(means[n, :, :] - sample))
        
    dens = tf.map_fn(
        fn=aux_f,
        elems=tf.range(n_samples),
        dtype=dType
    )
    
    return dens
    
    
    
    
    
def all_(flatted_means, flatted_samples, proposal_settings):
    """
    Calculate the total denominator for each sample.

    Calculate the total denominator as the average of the density evaluation
    of all the proposals on each sample.

    .. math::
    
        \Phi(x_{n,t}) = \dfrac{1}{N} \dfrac{1}{T}\sum_{i=1}^N \sum_{k=1}^T q(x_{i,t} | \mu_{i, k})
        
    Parameters
    ----------
    flatted_means : tensorflow.Tensor, shape (N*T, dim)
        The tensor of means flattened from (N, T, dim) to (N*T, dim).
    flatted_samples : tensorflow.Tensor, shape (N*T*M, dim)
        The tensor of samples flattened from (N, T*M, dim) to (N*T*M, dim).
    proposal_settings : dict
        Dictionary containing the proposal settings, including the covariance matrix and the proposal type. The possible
        keys for this dictionary are:
        
        - "cov": the covariance matrix of the proposal distribution.
        - "proposal_type": the type of proposal distribution. Possible values are "gaussian" and "student".
        - "df": the degrees of freedom of the student-t distribution. Only used if "proposal_type" is "student".

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