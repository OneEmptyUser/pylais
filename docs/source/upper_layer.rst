The MCMC layer
==============

In the first part of LAIS the location parameter proposal densities are adapted to the invariant density. The adaptation is
perform through ``N`` markov chains. By default, the invariant density of every chain is the posterior density of the model.
Other options can be:

#. Different posterior considering the same likelihood but different priors.

#. Tempered versions of the posterior distribution.

#. Posteriors built using a subset of the data, i.e., partial posteriors.


This adaptation
is performed with the ``upper_layer`` method.

.. automethod:: pylais.Lais.upper_layer

The parameters of the MCMC methods used can be changed passing a dictionary to the argument ``settings``.
The possible argument of each method are:

#. Random walk Metropolis Hastings.
    - cov: covariance matrix

#. Hamiltonian Monte Carlo.
    - step_size
    - num_leapfrog_steps

#. No-U-Turn Sampler.
    - step_size.

#. Metropolis Adjusted Langevin Algorithm.
    - step_size

#. Slice Sampling.
    - step_size.
    - max_doublings

.. code-block::python

    settings = {"step_size": 0.01, "num_leapfrog_steps": 10}

Different targets
-----------------

If we want to use different invariant densities for each chain we have to pass a list of functions to
the argument ``targets`` of ``Lais.upper_layer``. This list of functions must have the same length as
the number of chains ``N``. If this argument is left as default, then all the chains will have the
posterior distribution of the model as invariant density.

.. code-block:: python

    T, N, dim = 1000, 3, 1
    def target_1(theta):
        ...
    def target_2(theta):
        ...
    def target_3(theta):
        ...
        
    my_lais.upper_layer(T=T, N=N, initial_points=tf.random.normal((N,dim)),
    targets=[target_1, target_2, target_3])




MCMC outputs
------------

After running the ``upper_layer`` method we will have **N** Markov chains where each state is the mean of a proposal density.
This output is an object of the class ``mcmcSamples``. This object can be instanciated only passing the ``samples`` argument, a tensor
of shape **(N, T, dim)**.

.. .. autofunction:: pylais.samples.mcmcSamples

Several methods in this class allows to assess the convergence of the Markov chains.

#. .. automethod:: pylais.samples.mcmcSamples.gelman_rubin
#. .. automethod:: pylais.samples.mcmcSamples.trace
#. .. automethod:: pylais.samples.mcmcSamples.scatter
#. .. automethod:: pylais.samples.mcmcSamples.autoCorrPlot
#. .. automethod:: pylais.samples.mcmcSamples.cumulativeMean