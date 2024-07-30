Examples
========

In this section we show how to use pylais in a few examples:

#. A simple integration problem.

#. A non-linear regression problem.

Simple integration problem
--------------------------

In this case we are going to integrate 

.. math::

    Z = \int_{\mathbb{R}^2} \dfrac{1}{2\pi\sqrt{\det(\Sigma)}} \exp\left(-\dfrac{1}{2}(x - \mu)^{\top}\Sigma^{-1}(x - \mu) \right) d{x},

with :math:`\mu=[-10, 10]` and :math:`\Sigma=\begin{pmatrix} 2 & 0.6 \\ 0.6 & 1 \end{pmatrix}`.

First we are going to define this function as the loglikelihood in a different file called `ex_integration.py`:

.. code:: python

    import tensorflow as tf
    import tensorflow_probability as tfp
    
    def logtarget(x):
        loc = tf.constant([-10, 10], dtype=tf.float64)
        cov = tf.constant([[2, 0.6], [0.6, 1]], dtype=tf.float64)
        pdf = tfp.distributions.MultivariateNormalTriL(loc,
                    scale_tril=tf.linalg.cholesky(cov))
        return pdf.log_prob(x)

After this we can use `pylais` to integrate.

.. code:: python

    from ex_integration import logtarget
    import tensorflow as tf
    from pylais import Lais

    # Instanciate the Lais class
    myLais = Lais(logtarget)

    # Define the parameters of the upper layer
    n_iter = 5000
    N = 3
    gen = tf.random.Generator.from_seed(1)
    initial_points = 2*gen.uniform((N, 2), dtype=tf.float64)
    mcmc_settings = {
        cov = tf.eye(dim, dtype=tf.float64)*0.1
    }
    method = "rw"

    mu_chains = myLais.upper_layer(n_iter=n_iter,
                               N=N,
                               initial_points=initial_points,
                               method=upper_settings,
                               mcmc_settings=mcmc_settings)

The chains can be explored for assessing the convergence with

.. code:: python

    mu_chains.trace() # plot the trace
    mu_chains.cumulativeMean() # plot the cumulative mean

After having convergence in the MCMC chain we can pass to the sampling procedure

.. code:: python

    lower_settings = {
    "cov": tf.eye(dim, dtype=tf.float64),
    "den": "all",
    "n_per_sample": 3
    }
    samples = myLais.lower_layer(cov=lower_settings["cov"],
                                 n_per_sample=lower_settings["n_per_sample"],
                                 den=lower_settings["den"])

After the lower layer is done we can see the value of the integral through the value the marginal likelihood, i.e., `Z`:

.. code:: python

    print(samples.Z)



Simple non-linear regression
----------------------------

In this case we work with generated data, for simplicity we define a class `ExampleReg` that generates the data at instaciation time
and also contains the loglikelihood and logprior, this code is save in the file `non_linear.py`:

.. code:: python

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

    @tf.function
    def logprior(self, theta):
        a = theta[0]
        b = theta[1]
        a_limits = tf.constant((0, 10), dtype=tf.float64)
        b_limits = tf.constant((0, 6), dtype=tf.float64)
        if (a_limits[0]<a and a<a_limits[1]) and (b_limits[0] < b and b < b_limits[1]):
            # return tf.constant(0, dtype=tf.float64)
            return tf.math.log(tf.constant(1/60, dtype=tf.float64))
        else:
            return tf.constant(-numpy.inf, dtype=tf.float64)

    def loglikelihood(self, theta):
        y_est = self.f(theta)
        return -tf.math.reduce_sum(tf.math.square(self.y - y_est))

Now we can use `pylais` to calculate the mean of the posterior distribution:

.. code-block:: python

    from non_linear import ExampleReg
    from pylais import Lais
    import tensorflow as tf

    example = ExampleReg()
    loglikelihood = example.loglikelihood
    logprior = example.logprior

    n_iter = 5000
    N = 3
    gen = tf.random.Generator.from_seed(1)
    initials_points = 2*gen.uniform((N, dim), dtype=tf.float64)
    method = "hmc"

    # Run the upper layer (MCMC layer)
    mcmc_settings = {
        "step_size": 0.01,
        "num_leapfrog_steps": 15,
        "max_doublings":10,
    }

    mu_chains = myLais.upper_layer(n_iter=n_iter,
                                N=N,
                                initial_points=initials_points,
                                method=method,
                                mcmc_settings=mcmc_settings)


    # Run the lower layer (IS layer)
    lower_settings = {
        "cov": 1e-2*tf.eye(dim, dtype=tf.float64),
        "den": "all",
        "n_per_sample": 3
    }

    final_samples = myLais.lower_layer(cov=lower_settings["cov"],
                                   n_per_sample=lower_settings["n_per_sample"],
                                   den=lower_settings["den"])
    expected = 	final_samples.moment_n().numpy()
