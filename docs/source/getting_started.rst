Getting started
===============

The main class of ``pylais`` package is called ``Lais``.


This class allows to define the problem and sample from target density.
Firs of all, for creating an instance of the class we need to provide the ``loglikelihood`` of the model, this is
the logarithm of the **likelihood** function and
the logarithm of the **prior**, ``logprior``, being this last argument optional. If no ``logprior`` is passed, then we work with the
improper prior distribution.

The ``loglikelihood`` function and ``logprior`` should be defined using tensorflow functions. Example of this can be

Example of a uniform prior in the :math:`[0, 10] \times [0, 2\pi]` and the log-likelihood of a guassian model
are presented below:

.. code-block:: python
    
    import tensorflow as tf
    from numpy import inf
    @tf.function
    def logprior(theta):
        a = theta[0]
        b = theta[1]
        a_limits = tf.constant((0, 10), dtype=tf.float64)
        b_limits = tf.constant((0, 2*pi), dtype=tf.float64)
        if (a_limits[0]<a and a<a_limits[1]) and (b_limits[0] < b and b < b_limits[1]):
            return tf.math.log(tf.constant(1/60, dtype=tf.float64))
        else:
            return tf.constant(-inf, dtype=tf.float64)

    def loglikelihood(theta):
        return -0.5*(tf.math.reduce_sum((theta - 1)**2))


After defining the ``loglikelihood`` and ``logprior`` function we create an instance of the class doing:

.. code-block:: python

    my_lais = Lais(loglikelihood, logprior)


Following we have a list of all the methods in ``Lais``:

.. autoclass:: pylais.lais.Lais