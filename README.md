# Layered adaptive importance sampling for python


Layered Adaptive Importance Sampling is an IS method that consists of two layers of sampling. In the upper layer, the location parameters of the proposal densities to be used in the lower layer are adapted. The adaptation is carried out by running $N$ MCMC chains of length $T$. After $T$ iterations, we have a population of $NT$ location parameters $\{\bmu_{n,t}\}$. In the lower layer, Importance Sampling is performed, where samples $\textbf{x}_{n,t}\sim q(\bx | \bmu_{n,t})$ are drawn, and each one is associated with a weight.


The main class of the `pylais` package is the class \verb*|Lais|. This class can be instantiated with the logarithm of the likelihood and the logarithm of the prior.

The main methods in this class are:
- upper_layer: for adapting the MCMC chains to the target distribution.
- lower_layer: for performing IS step
- main: runs sequentially the upper layer and then le lower layer.

## Examples
As examples of the use of pylais we present a simple integration problem and a non-linear regression one.

### Integration problem
```{python}

import tensorflow as tf
import tensorflow_probability as tfp
from pylais import Lais

def logtarget(x):
    loc = tf.constant([-10, 10], dtype=tf.float64)
    cov = tf.constant([[2, 0.6], [0.6, 1]], dtype=tf.float64)
    # pdf = tfp.distributions.MultivariateNormalFullCovariance(loc, cov)
    pdf = tfp.distributions.MultivariateNormalTriL(loc, scale_tril=tf.linalg.cholesky(cov))
    return pdf.log_prob(x)

dim = 2
N = 3
n_iter = 5000
cov = tf.eye(dim, dtype=tf.float64)*0.01
n_per_sample = 1
den = "all"
method = "hmc"
settings = {"step_size": 0.01, "num_leapfrog_steps": 10,
            "max_doublings":10, "cov": cov}

myLais = Lais(logtarget)

gen = tf.random.Generator.from_seed(1)
initial_points = gen.uniform((N, dim), dtype=tf.float64)

means = myLais.upper_layer(n_iter, N, initial_points, method=method, mcmc_settings=settings)
ImpSamples = myLais.lower_layer(cov, n_per_sample, den)
print(ImpSamples.Z)
```