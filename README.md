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
In this problem we want to calculate the integral the marginal likelihood of a function.
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

### Non-linear regression
In this example we have data that comes from the function
$f(t|\theta) = \exp{(-\theta_0t)}\sin{(\theta_1t)}$. Let's simulate the data and save everything in a class.
```
class ExampleReg:
    A, B = 0.1, 2
    theta_true = tf.constant((A, B), dtype=tf.float64)
    def __init__(self):
        self.x = tf.linspace(0, 10, 50)
        self.y = self.f(self.theta_true) + tf.random.normal(shape=(50,), stddev=0.1, dtype=tf.float64)

    def f(self, theta):
        a = theta[0]
        b = theta[1]
        return tf.exp(-a * self.x)*tf.sin(b * self.x)
    
    def loglikelihood(self, theta):
        y_est = self.f(theta)
        return -tf.math.reduce_sum(tf.math.square(self.y - y_est))

    def logprior(theta):
    a = theta[0]
    b = theta[1]
    if (0<a and a<10) and (0 < b and b < 6):
        return 0
    else:
        return -tf.constant(numpy.inf)
```
After having our likelihood and prior we use lais to estimate the mean of the posterior distribution.
```
example = ExampleReg()

logtarget = example.loglikelihood
logprior = example.logprior
myLais = Lais(logtarget, logprior)

gen = tf.random.Generator.from_seed(1)
initial_points = gen.uniform((N, dim), dtype=tf.float64)

upper_settings = {"method": method, "mcmc_settings": settings}
lower_settings = {"cov": cov, "den": den, "n_per_sample": n_per_sample}
samples = myLais.main(n_iter, N, initial_points, upper_settings, lower_settings)
print(samples.moment_n())
```