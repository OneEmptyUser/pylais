# Layered adaptive importance sampling for python


Layered Adaptive Importance Sampling is an IS method that consists of two layers of sampling. In the upper layer, the location parameters of the proposal densities to be used in the lower layer are adapted. The adaptation is carried out by running $N$ MCMC chains of length $T$. After $T$ iterations, we have a population of $NT$ location parameters $\{\bmu_{n,t}\}$. In the lower layer, Importance Sampling is performed, where samples $\bx_{n,t}\sim q(\bx | \bmu_{n,t})$ are drawn, and each one is associated with a weight.


The main class of the ´pylais´ package is the class \verb*|Lais|. This class can be instantiated with the logarithm of the likelihood and the logarithm of the prior. This second argument can be left out, in that case the assumed prior will be the improper uniform prior, i.e., $g(\bx) = 1$.