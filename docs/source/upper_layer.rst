The MCMC layer
==============

In the first part of LAIS the proposal densities are adapted to the target density, i.e., the posterior. This adaptation
is performed with the ``upper_layer`` method.

.. automethod:: pylais.Lais.upper_layer

This method can adapt more than one chain to the target distribution as long as we pass **N** initial points.
This points must have the shape **(N, dim)** with **dim** being the dimension of the parameter space.

Example:


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