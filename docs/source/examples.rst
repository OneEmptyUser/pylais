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
    N = 4
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

.. figure:: _static\\trace_ex_1.png

.. code:: python

    mu_chains.cumulativeMean() # plot cumulative mean

.. figure:: _static\\cummulativeMean_ex_1.png

After having explore the MCMC chain we can pass to the sampling procedure. Remember that for LAIS algorithm
is not strictly required that the chains converge, but it can beneficial.

.. code:: python

    lower_settings = {
    "cov": tf.eye(dim),
    "den": "all",
    "M": 1
    }
    samples = myLais.lower_layer(cov=lower_settings["cov"],
                                 M=lower_settings["M"],
                                 den=lower_settings["den"])

After the lower layer is done we can see the value of the integral through the value the marginal likelihood, i.e., `Z`:

.. code:: python

    print(samples.Z)
    >>> <tf.Tensor: shape=(), dtype=float64, numpy=0.9821657584812903>

We can explore the sampling procedure with a histogram of resampled samples. This can be done with the method ``histogram``. To see the
histogram only of the marginal of :math:`\theta_1`:

.. code-block:: python

    myLais.histogram(5000, dimension=(1,))

.. figure:: _static\\histogram_una_dim_ex_1.png

or the histogram in the complete space:

.. code-block:: python

    myLais.histogram(5000, dimension=(0, 1))

.. figure:: _static\\histogram_ex_1.png


Simple non-linear regression
----------------------------

In this case, our objective will be the estimation of the parameters of a model given by the Equation \eqref{equ:regression-example}.

.. math::

	y_i = \exp(-\alpha t_i) \sin(\beta t_i) + v_i, \qquad v_i \sim \mathcal{N}(0, 0.1^2).

We work with data generated from this model, for simplicity we define a class `ExampleReg` that generates the data at instantiation time
and also contains the ``loglikelihood`` and ``logprior``, this code is save in the file `non_linear.py`:

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
        return (-(1)/(2*0.1**2))*tf.math.reduce_sum(tf.math.square(self.y - y_est))

Now we can use `pylais` to calculate the mean of the posterior distribution. We use Hamiltonian Monte Carlo in the upper layer and
a Student-t proposal with five degrees of freedom in the lower layer.


.. code-block:: python

    from non_linear import ExampleReg
    from pylais import Lais
    import tensorflow as tf

    example = ExampleReg()
    loglikelihood = example.loglikelihood
    logprior = example.logprior

    # Define the settings
    N = 4
    T = 5000
    gen = tf.random.Generator.from_seed(1)
    initial_points = 2*gen.uniform((N, 2), dtype=tf.float64)

    upper_settings = {"method": "hmc", 
        "settings": {"step_size": 0.01, "num_leapfrog_steps": 10}}

    lower_settings = {"cov": tf.eye(2)*0.01, "den": "all", "M": 1,
        "proposal_type": "student", "df": 5}

    # Instanciate the class
    my_lais = Lais(loglikelihood=loglikelihood, logprior=logprior)

    # Run the MCMC layer
    mu_chains = my_lais.upper_layer(T=T, N=N, initial_points=initials, 
        method=upper_settings["method"], settings=upper_settings["settings"])

    # Run the IS layer
    final_samples = my_lais.lower_layer(cov=lower_settings["cov"], M=lower_settings["M"],
        den=lower_settings["den"], proposal_type=lower_settings["proposal_type"], df=lower_settings["df"])

    # Calculate the expected value
    print(final_samples.moment_n())
    >>> <tf.Tensor: shape=(1, 2), dtype=float64, numpy=array([[0.10242369 1.99232299]])>


Partial posteriors
------------------

The term partial posteriors refers to posterior distributions built with a subset of the data. In this case the
chains in the upper layer will be targeting different partial posteriors, whilst the inference is done with respect
to the true posterior, this is, the weights in the lower layer are calculate using the total posterior. We use the
class ``ExampleReg`` created in the previous example. First, let us define a function that return a list with the partial
posteriors.

.. code-block:: python

    def new_targets(example):
	from copy import copy
	n_data = 5
	targets = []
	for n in range(N):
		idx =tf.random.categorical(tf.math.log([tf.ones(len(example.x))]), n_data)
		new_ex = copy(example)
		new_ex.x = tf.gather(new_ex.x, tf.squeeze(idx))
		new_ex.y = tf.gather(new_ex.y, tf.squeeze(idx))
		
		log_tar = lambda theta:new_ex.loglikelihood(theta) + new_ex.logprior(theta)
		targets.append(log_tar)
	return targets

Now lets estimate the expected value of the posterior distribution with LAIS using a different invariant density for each
chain.

.. code-block:: python

    targets_list = new_targets(example)

    myLais = Lais(loglikelihood, logprior)
    # Declare the initial points
    gen = tf.random.Generator.from_seed(1)
    initial_points = 2*gen.uniform((N, 2), dtype=tf.float64)

    upper_settings = {"method": "rwmh",
        settings={"cov": 0.1*tf.eye(2)}, "targets": targets_list}
    lower_settings = {"cov": 0.1*tf.eye(2), "M": 1, "den": "all",
        "proposal_type": "gaussian"}

    final_samples = myLais.main(T, N, initial_points, upper_settings, lower_settings)
    print(final_samples.momnet_n())
    >>> tf.Tensor([[0.1070357 2.01090445]], shape=(1, 2), dtype=float64)
