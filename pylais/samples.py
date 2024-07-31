import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import corrcoef
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow_probability as tfp
# from tfp.stats import correlation

class mcmcSamples:
    """Class for MCMC samples.
    
 
    Attributes
    ----------
    samples : tensorflow.Tensor 
        A tensor of shape (n_chains, n_iter, n_params) containing the samples

    Methods
    -------
    __str__(self)
    gelman_rubin(self)
        Calculate the Gelman-Rubin statistic for  the chains.
    trace(self, axis=None)
        Plot the trace of the chains.
    scatter(self, xlim=None, ylim=None, axis=None)
        Do an scatter plot of all the chains.
    autoCorrPlot(self, chain=0, component=0, max_lag=10, axis=None, plot_args={})
        Plot the autocorrelation function for the given chain and component.
    cummulativeMean(self)
        Plot the cumulative mean of the samples.
    """
    
    def __init__(self, samples):
        """
        Initializes a new instance of the class with the given samples.

        Parameters:
            samples (list): A list of samples.

        Returns:
            None
        """
        self.samples = samples

    def __str__(self):
        return f"{len(self.samples)} MCMC chains.\nGelman-Rubin statistic: {self.gelman_rubin()}"

    def gelman_rubin(self):
        """
        Calculate the Gelman-Rubin statistic for the chains.

        This function calculates the Gelman-Rubin statistic for the Markov chains stored in the `samples` attribute. The Gelman-Rubin statistic is a measure of convergence of multiple chains. It is calculated as follows:

        1. Calculate the standard deviation of each chain along the first axis (axis=1).
        2. Calculate the mean of the standard deviations along the first axis (axis=0).
        3. Calculate the standard deviation of the means along the first axis (axis=0).
        4. Calculate the Gelman-Rubin statistic as:
           ((n_iter - 1)/n_iter)*w + b/n_iter)/w

        where w is the mean of the standard deviations, b is the standard deviation of the means, and n_iter is the number of iterations in each chain.

        Parameters:
            None

        Returns:
            tf.Tensor: A tensor containing the Gelman-Rubin statistic. If there is only one chain, it prints a message saying to run more than one chain.
        """
        n_chains, n_iter, _ = self.samples.shape
        samples = self.samples
        if n_chains > 1:
            std = tf.math.reduce_std(samples, axis=1)
            w = tf.math.reduce_mean(std, axis=0)

            means = tf.math.reduce_mean(samples, axis=1)
            b = tf.math.reduce_std(means, axis=0)
            gr = (((n_iter - 1)/n_iter)*w + b/n_iter)/w; gr
        
            return gr
        else:
            print("Run more than one chain.")
        
    def trace(self, chains=None, axis=None):
        """
        Plot the traces of MCMC chains for selected dimensions.

        This function plots the traces of MCMC chains for the specified dimensions.
        It takes an optional parameter `chains` which specifies the chains to plot. If not provided,
        all chains will be plotted. The `chains` parameter can be an integer, a list, or a tuple.
        If an integer is provided, only the chain with the corresponding index will be plotted. If a list
        or tuple is provided, only the chains with the corresponding indices will be plotted. The function creates
        a new axis if `axis` is not provided. The function plots the traces of MCMC chains for the specified
        chain and returns None. In the legend the subscript stands for the component and the superscript for the chain.
        
        Parameters
        ----------
        chains : int, list, tuple, optional
            The chains to plot. If not provided, all chains will be plotted.
            - If an integer is provided, only the chain with the corresponding index will be plotted.
            - If a list or tuple is provided, only the chains with the corresponding indices will be plotted.
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            The axis on which to plot the traces. If not provided, a new axis will be created.

        Returns
        -------
            None
        """
        if axis:
            ax = axis
        else:
            _, ax = plt.subplots()

        if chains is None:
            chains_to_plot = range(self.samples.shape[0])
        else:
            if isinstance(chains, int):
                chains_to_plot = [chains]
            elif isinstance(chains, (list, tuple)):
                chains_to_plot = chains
            else:
                raise ValueError("Chains must be an int, a list or a tuple")
        # n_chains = self.samples.shape[0]
        
        labels = []
        for idim in range(self.samples.shape[2]):
            for n in chains_to_plot:
                ax.plot(self.samples[n, :, idim])
                labels.append(r"$\theta_{}^{}$".format(idim, n))
        ax.legend(labels, ncol=self.samples.shape[2], draggable=True)
        ax.set_title(f"Traces of {len(chains_to_plot)} MCMC chains")
        ax.set_xlabel("Iteration")
        plt.show()
        
    def autoCorrPlot(self,chain=0, component=0, max_lag=10, axis=None, plot_args={}):
        """
        Plot the autocorrelation of a given component of the MCMC samples.

        This function plots the autocorrelation of a given dimension of the MCMC samples and a given chain.
        The autocorrelation is calculated for a range of lags up to a maximum specified by the `max_lag` parameter.
        The resulting autocorrelation values are plotted as a stem plot.
        
        Parameters
        ----------
        chain : int, optional
            The index of the chain to plot the autocorrelation for. Defaults to 0.
        component : int, optional
            The index of the component to plot the autocorrelation for. Defaults to 0.
        max_lag : int, optional
            The maximum lag to calculate the autocorrelation up to. Defaults to 10.
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            The axis on which to plot the autocorrelation. If not provided, a new axis will be created.
        plot_args : dict, optional
            Additional arguments to pass to the `ax.stem` function. Defaults to an empty dictionary.

        Returns
        -------
            None
        """        
        
        if axis:
            ax = axis
        else:
            _, ax = plt.subplots()
        tensor = self.samples[chain, :, component]
        
        lags = range(1, min(tensor.shape[0]-10, max_lag))
        def calculateCorrelation(thin):
            tensor1 = tensor[:-thin]
            tensor2 = tensor[thin:]
            correlation = tfp.stats.correlation(tensor1, tensor2, event_axis=None).numpy()
            return correlation

        correlations = list(map(calculateCorrelation, lags))
        ax.stem(lags, correlations, **plot_args)
        ax.set_title(f"Autocorrelation plot of chain {chain} component {component}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")
        plt.show()
        
    
    def cumulativeMean(self, chains=None, axis=None):
        """
        Plot the cumulative means of the selected MCMC chains.

        This function plots the cumulative means of MCMC chains for the specified dimensions.
        It takes an optional parameter `chains` which specifies the chains to plot. If not provided,
        all chains will be plotted. The `chains` parameter can be an integer, a list, or a tuple, with
        the indices of the chains to plot. The function creates a new axis if `axis` is not provided.
        In the legend the subscript stands for the component and the superscript for the chain.

        Parameters
        ----------
        chains : int, list, tuple, optional
            The chains to plot. If not provided, all chains will be plotted.
            - If an integer is provided, only the chain with the corresponding index will be plotted.
            - If a list or tuple is provided, only the chains with the corresponding indices will be plotted.
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            The axis on which to plot the cumulative means. If not provided, a new axis will be created.

        Raises
        ------
        ValueError
            If the `chains` parameter is not an integer, a list, or a tuple.
            If a chain index is out of range.

        Returns
        -------
        None
        """

        if axis:
            ax = axis
        else:
            _, ax = plt.subplots()
        
        n_chains, iterations, dim = self.samples.shape
        
        if not chains:
            chains_to_plot = range(n_chains)
        else:
            if isinstance(chains, int):
                chains_to_plot = [chains]
            elif isinstance(chains, (list, tuple)):
                chains_to_plot = chains
            else:
                raise ValueError("Chains must be an int, a list or a tuple")
        dType = self.samples.dtype
        
        
        labels = []
        for idim in range(dim):
            for n in chains_to_plot:
                ax.plot(tf.math.cumsum(self.samples[n, :, idim])/tf.range(1, iterations+1,
                                                                            dtype=dType))
                labels.append(r"$\theta_{}^{}$".format(idim, n))
        ax.set_title("Cumulative mean of {} MCMC chains".format(len(chains_to_plot)))
        ax.legend(labels, ncol=dim, draggable=True)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cumulative mean")
        plt.show()
    
    def scatter(self, xlim=None, ylim=None, axis=None):
        """
        Plot the samples as a scatter plot.

        This function plots the samples as a scatter plot using matplotlib. The samples are plotted as points in a 2D space, so
        if the dimension of the samples is larger that 2, only the first two dimensions are plotted.

        Parameters:
            xlim: tuple, (optional)
                The x-axis limits for the plot. Defaults to None.
            ylim: tuple, (optional)
                The y-axis limits for the plot. Defaults to None.
            axis: matplotlib.axes.Axes, (optional)
                The axis object to plot on. Defaults to None.

        Returns:
            None
        """
        if axis:
            ax = axis
        else:
            _, ax = plt.subplots()
        if self.samples.ndim == 3:
            for i in range(len(self.samples)):
                ax.scatter(self.samples[i, :, 0], self.samples[i, :, 1], s=2)
        if self.samples.ndim == 2:
            ax.scatter(self.samples[:, 0], self.samples[:, 1], s=2)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        plt.show()
    
class ISSamples:
    """Class for Importance Sampling samples.
    
 
    Attributes
        samples : tensorflow.Tensor
            A tensor of shape (n_chains, n_iter, n_params)
        containing the samples.
        weights : tensorflow.Tensor
            A tensor of shape (n_chains, n_iter)
        normalized_weights : tensorflow.Tensor
            A tensor of shape (n_chains, n_iter) equals to weights / tf.reduce_sum(weights)
    
    Methods:
        __str__(self)
        resample(self, n)
            Resample the samples.
        ess(self)
            Calculate the effective sample size.
        Z(self)
            Calculate the mean value of the weights.
        moment_n(self, n=1)
            Calculate the n-th moment of the samples.
        expected_f(self, f)
            Calculate the expected value of the function f evaluated at the samples.
    """
    def __init__(self, samples, weights):
        """
        Initializes the ISSamples object with the given samples and weights.

        Parameters:
            samples: (tensorflow.Tensor)
                A tensor of shape (n_chains, n_iter, n_params) containing the samples.
            weights: (tensorflow.Tensor)
                A tensor of shape (n_chains, n_iter) containing the weights.

        Returns:
            None
        """
        self.samples = samples
        self.weights = weights
        self.normalized_weights = weights / tf.math.reduce_sum(weights)
        
    def resample(self, n):
        """
        Resamples the samples based on the given number of samples.

        Parameters
        ----------
            n : int
                The number of samples to resample.

        Returns
        -------
            A tensor containing the resampled samples.
        """
        norm_weights = self.normalized_weights
        idx = tf.random.categorical(tf.math.log([norm_weights]), n)
        return tf.gather(self.samples, tf.squeeze(idx))
    
    @property
    def ess(self):
        """
        Calculate the effective sample size (ESS) of the samples.

        Returns:
            float: The effective sample size.
        """
        return 1 / tf.math.reduce_sum(tf.math.square(self.normalized_weights))
    @property
    def Z(self):
        """
        Calculate the mean value of the weights.

        Returns:
            float: The mean value of the weights.
        """
        return tf.math.reduce_mean(self.weights)
    
    def moment_n(self, n=1):
        """
        Calculate the expected value of the samples raised to the power of `n`.

        Parameters
        ----------
            n: int (optional)
                The power to raise the samples to. Defaults to 1.

        Returns
        -------
            The expected value of the samples raised to the power of `n`.
        """
        samples = self.samples
        # flatted_normalized_weights = self.normalized_weights
        norm = self.normalized_weights[tf.newaxis, :]
        expected = tf.matmul(norm, samples**n)
        return expected
    
    def expected_f(self, f):
        """
        Calculate the expected value of the function `f` applied to the samples.

        Parameters:
            f: (function)
                The function to be applied to the samples.

        Returns:
            The expected value of `f` applied to the samples.
        """
        samples = self.samples
        norm = self.normalized_weights[tf.newaxis, :]
        f_values = tf.vectorized_map(f, samples)[:, tf.newaxis]
        expected = tf.matmul(norm, f_values)
        return tf.squeeze(expected)
    
    def __str__(self):
        return f"{len(self.weights)} Samples class with ESS = {self.ess} and Z = {self.Z}"