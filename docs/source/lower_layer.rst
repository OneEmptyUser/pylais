The IS layer
============


Once the MCMC layer is run we can have the sampling procedure. In this case the samples of the adapted chains are used as
location parameters of the proposals densities, :math:`q_{t,n}`. From each of this densities we will sample :math:`M` samples
(named ``n_per_sample`` in the code). To each new sample, :math:`x_{t,n}^m`, is associated to an Importance Sampling weight,
such weight is calculated following the usual Adaptive Importance Sampling formula:

.. math::

    w_{t,n}^m = \dfrac{\pi(x_{t,n}^m)}{\Phi(x_{t,n}^m)}

There are three options for the function :math:`\Phi` to be calculated:

1. The **spatial** denominator. In this case the :math:`\Phi` is given by:

.. autofunction:: pylais.denominators.spatial

2. The **temporal** denominator. 

.. autofunction:: pylais.denominators.temporal

3. The **total** denominator.

.. autofunction:: pylais.denominators.all_

Importance Sampling outputs
---------------------------

When the sampling process is over we have a cloud of particles with the associated weights that allows
for the approximation of target density, i.e., the posterior. For simplicity the output of the method
``lower_layer`` is an object of class ``ISSamples``. This class presents three attributes:

#. ``samples``: the samples drawn from the proposals.
#. ``weights``: the associated importance weight.
#. ``normalized_weights``: the weights normalized to sum up to 1.

In this manner we can easily approximate the target distribution through a resampling of the 
sampled samples according to its weight, we can do that with the ``resample`` method.

.. autofunction:: pylais.samples.ISSamples.resample

Besides there exists methods to calculate quantities of interest as the marginal likelihood and the moments of
the posterior:

#. .. py:property:: pylais.samples.ISSamples.Z

      Return the value of marginal likelihood of the posterior


.. #. .. autofunction:: pylais.samples.ISSamples.moment_n
.. #. .. autofunction:: pylais.samples.ISSamples.expected_f