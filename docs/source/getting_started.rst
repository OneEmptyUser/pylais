Getting started
===============

The main class of ``pylais`` package is the ``Lais``.

This class allows to define the problem and sample from target density.
Firs of all, for instanciating the class we need to provide the ``loglikelihood`` of the model and
the logarithm of the prior, ``logprior``, being this last argument optional.

.. .. autofunction:: pylais.Lais.__init__

The ``loglikelihood`` function and ``logprior`` should be defined using tensorflow functions. Example of this can be

Example of a uniform prior in the :math:`[0, 10] \times [0, 2\pi]`.

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
            # return tf.constant(0, dtype=tf.float64)
            return tf.math.log(tf.constant(1/60, dtype=tf.float64))
        else:
            return tf.constant(-inf, dtype=tf.float64)


