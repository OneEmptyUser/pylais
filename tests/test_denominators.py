import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # THIS IS TO REMOVE A WARNING
import sys
# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pylais')))
import tensorflow as tf
import tensorflow_probability as tfp
from denominators import all_, spatial, temporal, spatial2
from utils import flatTensor3D, repeatTensor3D

mvn = tfp.distributions.MultivariateNormalFullCovariance(
    covariance_matrix=tf.eye(2, dtype=tf.float64)
)
tf.random.set_seed(27)


N = 3
n_iter = 10
n_per_sample = 2
dim = 2

fake_means = []
for i in range(N):
  matrix = [tf.zeros(2, dtype=tf.float64)]
  for t in range(n_iter-1):
    matrix.append(mvn.sample() + matrix[-1])
  tensor = tf.stack(matrix)
  tensor = tf.expand_dims(tensor, axis=0)
  fake_means.append(tensor)
fake_means = tf.concat(fake_means, axis=0)

repeated_means = repeatTensor3D(fake_means, n_per_sample)
flatted_repeated_means = flatTensor3D(repeated_means)
flatted_means = flatTensor3D(fake_means)
mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(2, dtype=tf.float64),
                                                            covariance_matrix=tf.eye(2, dtype=tf.float64))

flatted_samples = mvn.sample(n_per_sample*n_iter*N) + flatted_repeated_means
samples = tf.reshape(flatted_samples, (N,n_iter*n_per_sample, dim))


def test_all_():
    
    cov = tf.constant([[1, 0.5],
                       [0.5, 1]], dtype=tf.float64)
    expected_denominators = []
    for n in range(flatted_samples.shape[0]):
        mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=flatted_samples[n],
                                                                 covariance_matrix=cov)
        expected_denominators.append(tf.math.reduce_mean(mvn.prob(flatted_means)).numpy())
    
    
    
    proposal_settings = {"proposal_type": "gaussian", "cov": cov}
    assert tf.reduce_all(tf.constant(expected_denominators) == all_(flatted_means, flatted_samples, proposal_settings))
    
def test_all_student():
    
    cov = tf.constant([[1, 0.5],
                       [0.5, 1]], dtype=tf.float64)
    scale = tf.linalg.cholesky(cov)
    df = 10
    expected_denominators = []
    for n in range(flatted_samples.shape[0]):
        mvt = tfp.distributions.MultivariateStudentTLinearOperator(df=df,
                                                                   loc=flatted_samples[n],
                                                                   scale=tf.linalg.LinearOperatorLowerTriangular(scale))
        expected_denominators.append(tf.math.reduce_mean(mvt.prob(flatted_means)).numpy())
    
    
    
    proposal_settings = {"proposal_type": "student", "cov": cov, "df": df}
    # assert tf.reduce_all(tf.constant(expected_weights) == all_(flatted_means, flatted_samples, proposal_settings))
    assert tf.reduce_all(abs(tf.constant(expected_denominators) - all_(flatted_means, flatted_samples, proposal_settings))<1e-10)
    
def test_temporal():
    cov = tf.constant([[1, 0.5],
                       [0.5, 1]], dtype=tf.float64)
    expected_denominators = []
    for n in range(samples.shape[0]):
        # weights_chain = []
        for t in range(samples.shape[1]):
            loc = samples[n, t, :]
            mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=loc,
                                                                     covariance_matrix=cov)
            expected_denominators.append(tf.math.reduce_mean(mvn.prob(fake_means[n, :, :])).numpy())
        # weights_chain = tf.stack(weights_chain)
        # expected_weights.append(weights_chain.numpy())
    proposal_settings = {"proposal_type": "gaussian", "cov": cov}
    assert tf.reduce_all(tf.constant(expected_denominators) == temporal(fake_means, samples, proposal_settings))
    
def test_temporal_student():
    cov = tf.constant([[1, 0.5],
                       [0.5, 1]], dtype=tf.float64)
    scale = tf.linalg.cholesky(cov)
    df = 10
    expected_denominators = []
    for n in range(samples.shape[0]):
        # weights_chain = []
        for t in range(samples.shape[1]):
            loc = samples[n, t, :]
            mvt = tfp.distributions.MultivariateStudentTLinearOperator(df=df,
                                                                       loc=loc,
                                                                       scale=tf.linalg.LinearOperatorLowerTriangular(scale))
            expected_denominators.append(tf.math.reduce_mean(mvt.prob(fake_means[n, :, :])).numpy())
        # weights_chain = tf.stack(weights_chain)
        # expected_weights.append(weights_chain.numpy())
    proposal_settings = {"proposal_type": "student", "cov": cov, "df": df}
    # assert tf.reduce_all(tf.constant(expected_weights) == temporal(fake_means, samples, proposal_settings))
    assert tf.reduce_all(abs(tf.constant(expected_denominators) - temporal(fake_means, samples, proposal_settings)) < 1e-10)
        

def test_spatial():
    cov = tf.constant([[1, 0.5],
                       [0.5, 1]], dtype=tf.float64)
    expected_denominators = []
    
    n_per_sample = samples.shape[1]//fake_means.shape[1]
    for n in range(samples.shape[0]):
        for t in range(samples.shape[1]):
            loc = samples[n, t, :]
            # mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=loc,
            #                                                          covariance_matrix=cov)
            mvn = tfp.distributions.MultivariateNormalTriL(loc=loc,
                                                           scale_tril=tf.linalg.cholesky(cov))
            expected_denominators.append(tf.math.reduce_mean(mvn.prob(repeated_means[:, t, :])).numpy())
        # weights_chain = tf.stack(weights_chain)
        # expected_weights.append(weights_chain.numpy())
    proposal_settings = {"proposal_type": "gaussian", "cov": cov}
    assert tf.reduce_all(tf.constant(expected_denominators) == spatial(fake_means, samples, proposal_settings))
    
def test_spatial_other():
    cov = tf.constant([[1, 0.5],
                       [0.5, 1]], dtype=tf.float64)
    expected_denominators = []
    n_per_sample = samples.shape[1]//fake_means.shape[1]
    N, n_samples, _ = samples.shape
    # assert N == 3
    for n in range(N):
        for t in range(n_samples):
            den = 0
            for iprop in range(N):
                # den += tfp.distributions.MultivariateNormalFullCovariance(
                #     loc=repeated_means[iprop, t, :],
                #     covariance_matrix=cov
                # ).prob(samples[n, t, :]).numpy()
                den += tfp.distributions.MultivariateNormalTriL(
                    loc=repeated_means[iprop, t, :],
                    scale_tril=tf.linalg.cholesky(cov)
                ).prob(samples[n, t, :]).numpy()
            expected_denominators.append(den/N)
        # weights_chain = tf.stack(weights_chain)
        # expected_weights.append(weights_chain.numpy())
    proposal_settings = {"proposal_type": "gaussian", "cov": cov}
    # assert tf.reduce_all(tf.constant(expected_weights) == spatial(fake_means, samples, proposal_settings))
    assert tf.reduce_all(tf.constant(expected_denominators) == spatial(fake_means, samples, proposal_settings))


def test_spatial_student():
    cov = tf.constant([[1, 0.5],
                       [0.5, 1]], dtype=tf.float64)
    expected_denominators = []
    scale = tf.linalg.cholesky(cov)
    df = 10
    n_per_sample = samples.shape[1]//fake_means.shape[1]
    for n in range(samples.shape[0]):
        for t in range(samples.shape[1]):
            loc = samples[n, t, :]
            mvt = tfp.distributions.MultivariateStudentTLinearOperator(df=df,
                                                                       loc=loc,
                                                                       scale=tf.linalg.LinearOperatorLowerTriangular(scale))
            expected_denominators.append(tf.math.reduce_mean(mvt.prob(repeated_means[:, t, :])).numpy())
        # weights_chain = tf.stack(weights_chain)
        # expected_weights.append(weights_chain.numpy())
    proposal_settings = {"proposal_type": "student", "cov": cov, "df": df}
    # assert tf.reduce_all(tf.constant(expected_weights) == spatial(fake_means, samples, proposal_settings))
    assert tf.reduce_all(abs(tf.constant(expected_denominators) - spatial(fake_means, samples, proposal_settings)) < 1e-10)

def test_spatial2():
    cov = tf.constant([[1, 0.5],
                       [0.5, 1]], dtype=tf.float64)
    expected_denominators = []
    
    for n in range(samples.shape[0]):
        for t in range(samples.shape[1]):
            loc = samples[n, t, :]
            # mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=loc,
            #                                                          covariance_matrix=cov)
            mvn = tfp.distributions.MultivariateNormalTriL(loc=loc,
                                                           scale_tril=tf.linalg.cholesky(cov))
            expected_denominators.append(tf.math.reduce_mean(mvn.prob(repeated_means[:, t, :])).numpy())
        # weights_chain = tf.stack(weights_chain)
        # expected_weights.append(weights_chain.numpy())
    proposal_settings = {"proposal_type": "gaussian", "cov": cov}
    dens = spatial2(fake_means, flatted_samples, proposal_settings)
    assert tf.reduce_all(expected_denominators == dens)
    
def test_spatial2_other():
    cov = tf.constant([[1, 0.5],
                       [0.5, 1]], dtype=tf.float64)
    proposal_settings = {"proposal_type": "gaussian", "cov": cov}
    dens1 = spatial(fake_means, samples, proposal_settings)
    dens2 = spatial2(fake_means, flatted_samples, proposal_settings)
    
    assert tf.reduce_all(dens1 == dens2)