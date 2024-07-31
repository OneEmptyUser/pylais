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
    
    def scatter(self, xlim=None, ylim=None, chains=None, axis=None, dims=(0, 1)):
        """
        Plot a scatter plot of the MCMC samples.
        
        The user can choose which chains to plot. By default all chains will be plotted.
        The first two dimensions are plotted by default, but the user can choose different dimensions
        changing the `dims` parameter. If an axis is not provided, a new one will be created.

        Parameters
        ----------
        xlim : tuple, optional
            The x-axis limits of the plot. Defaults to None.
        ylim : tuple, optional
            The y-axis limits of the plot. Defaults to None.
        chains : int, list, tuple, optional
            The chains to plot. If not provided, all chains will be plotted.
            - If an integer is provided, only the chain with the corresponding index will be plotted.
            - If a list or tuple is provided, only the chains with the corresponding indices will be plotted.
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            The axis on which to plot the scatter plot. If not provided, a new axis will be created.
        dims : tuple, optional
            The dimensions of the samples to plot. Defaults to (0, 1).

        Raises
        ------
        ValueError
            If `chains` is not an integer, a list, or a tuple.
        ValueError
            If trying to access a chain or dimension that does not exist.

        Returns
        -------
        None
        """
        
        
        # If axis is provided, use it, otherwise create a new one.
        if axis:
            ax = axis
        else:
            _, ax = plt.subplots()
        
        # If not chains is provided, all chains will be plotted.
        if not chains:
            chains_to_plot = range(self.samples.shape[0])
        else:
            # If chains is an integer, only the chain with the corresponding index will be plotted.
            if isinstance(chains, int):
                chains_to_plot = [chains]
            # If chains is a list or tuple, only the chains with the corresponding indices will be plotted.
            elif isinstance(chains, (list, tuple)):
                chains_to_plot = chains
            # If chains is not an integer, a list, or a tuple, an error will be raised.
            else:
                raise ValueError("Chains must be an int, a list or a tuple")
            
        labels = []
        # If the samples are 3D, each chain will be plotted as a scatter plot.
        if self.samples.ndim == 3:
            for n in chains_to_plot:
                try:
                    ax.scatter(self.samples[n, :, dims[0]], self.samples[n, :, dims[1]], s=2)
                except Exception as e:
                    raise(ValueError(f"Trying to access a chain or dimension that does not exist: {e}"))
                labels.append("Chain {}".format(n))
        # If the samples are 2D, plot that chain
        if self.samples.ndim == 2:
            ax.scatter(self.samples[:, dims[0]], self.samples[:, dims[1]], s=2)
        
        # Set the limits of the plot, if provided.
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
            
        ax.set_title("Scatter plot of {} chains".format(len(chains_to_plot)))
        ax.legend(labels, ncol=1, draggable=True)
        ax.set_xlabel(r"$\theta_{}$".format(dims[0]))
        ax.set_ylabel(r"$\theta_{}$".format(dims[1]))
        plt.show()
    
class ISSamples:
    """Class for Importance Sampling samples.
    
 
    Attributes
    ----------
        samples : tensorflow.Tensor
            A tensor of shape (n_chains, n_iter, n_params)
        containing the samples.
        weights : tensorflow.Tensor
            A tensor of shape (n_chains, n_iter)
        normalized_weights : tensorflow.Tensor
            A tensor of shape (n_chains, n_iter) equals to weights / tf.reduce_sum(weights)
    
    Methods
    -------
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
        Initialize an instance of the class.

        Parameters
        ----------
        samples : tf.Tensor
            The samples to be stored in the instance.
        weights : tf.Tensor
            The weights corresponding to the samples.

        Returns
        -------
        None
        """
        self.samples = samples
        self.weights = weights
        self.normalized_weights = weights / tf.math.reduce_sum(weights)
        self._index = 0
    
    def __len__(self):
        """
        Return the number of samples in the instance.

        Returns
        -------
        int
            The number of samples in the instance.
        """
        return len(self.samples)
    
    def __getitem__(self, key):
        """
        Get an item from the ISSamples object.

        Parameters
        ----------
        key : int or slice
            The index or slice of the samples to retrieve.

        Returns
        -------
        tuple
            A tuple containing the sample and the corresponding normalized weight.
        """
        
        return self.samples[key], self.normalized_weights[key]
    
    def __iter__(self):
        """
        Initialize the iterator and set the internal index to 0.

        Returns
        -------
        self : Iterator
            The iterator object.
        """
        
        self._index = 0
        return self
    
    def __next__(self):
        """
        Return the next item from the iterator.

        This method is part of the iterator protocol and is called by the built-in function `next()`.
        It returns the next item from the iterator and increments the internal index.
        If the internal index is less than the length of the iterator, it returns the item at the current index
        and increments the index. Otherwise, it raises a `StopIteration` exception.

        Returns
        -------
        Any
            The next item from the iterator.

        Raises
        ------
        StopIteration
            If the internal index is equal to or greater than the length of the iterator.
        """
        if self._index < len(self):
            result = self[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
    
    def __str__(self):
        """
        Return a string representation of the ISSamples object.

        Returns
        -------
        str
            A string representation of the ISSamples object, including the number of samples, the estimated
            effective sample size (ESS), and the Z-value.

        """
        return f"{len(self.weights)} Samples class with ESS = {self.ess} and Z = {self.Z}"
    
    def resample(self, n, seed=None):
        """
        Resample the IS samples.

        This function resamples the IS samples by randomly selecting `n` samples from the current set of samples.
        The resampling is done based on the normalized weights of the samples. The function returns a new instance
        of the ISSamples class containing the resampled samples. The weights of the new samples are set to 1.

        Parameters
        ----------
        n : int
            The number of samples to resample.
        seed : int, optional
            The random seed to set before resampling. If not provided, no seed is set.

        Returns
        -------
        ISSamples
            A new instance of the ISSamples class containing the resampled samples.
        """
        
        if seed:
            tf.random.set_seed(seed)
            
        norm_weights = self.normalized_weights
        idx = tf.random.categorical(tf.math.log([norm_weights]), n)
        new_samples = tf.gather(self.samples, tf.squeeze(idx))
        new_weights = tf.ones(n)
        resampled_samples = ISSamples(new_samples, new_weights)
        return resampled_samples
    
    def scatter(self, xlim=None, ylim=None, axis=None, dims=(0, 1)):
        """
        Plot a scatter plot of the IS samples.

        By default, the function plots the first two dimensions of the samples, but the user can change
        this by providing the `dims` parameter.

        Parameters
        ----------
        xlim : tuple, optional
            The x-axis limits of the plot.
        ylim : tuple, optional
            The y-axis limits of the plot.
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            The axis on which to plot the scatter plot. If not provided, a new axis will be created.
        dims : tuple, optional
            The dimensions of the samples to plot. Defaults to (0, 1).

        Raises
        ------
        ValueError
            If trying to access a chain or dimension that does not exist.

        Returns
        -------
        None
        """
        
        
        if axis:
            ax = axis
        else:
            _, ax = plt.subplots()
        
        try:
            ax.scatter(self.samples[:, dims[0]], self.samples[:, dims[1]], s=2)
        except Exception as e:
            raise(ValueError(f"Trying to access a chain or dimension that does not exist: {e}"))
        
        # Set the limits of the plot, if provided.
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
            
        ax.set_title("Scatter plot of the IS samples")
        ax.set_xlabel(r"$\theta_{}$".format(dims[0]))
        ax.set_ylabel(r"$\theta_{}$".format(dims[1]))
        plt.show()
    @property
    def ess(self):
        """
        Compute the effective sample size (ESS) of the Importance Sampling samples.

        Returns
        -------
        float
            The effective sample size.
        """
        return (1 / tf.math.reduce_sum(tf.math.square(self.normalized_weights))).numpy()
    @property
    def Z(self):
        """
        Calculate the marginal likelihood.
        
        The marginal likelihood as defined in IS literature can be calculated as
        the mean of the importance weights.

        Returns
        -------
        tf.Tensor
            The mean of the `weights` attribute.
        """
        return tf.math.reduce_mean(self.weights)
    
    def moment_n(self, n=1):
        """
        Calculate the n-th moment of the Importance Sampling samples.

        Parameters
        ----------
        n : int, optional
            The order of the moment. Default is 1.

        Returns
        -------
        tf.Tensor
            The n-th moment of the samples.
        """
        samples = self.samples
        # flatted_normalized_weights = self.normalized_weights
        norm = self.normalized_weights[tf.newaxis, :]
        expected = tf.matmul(norm, samples**n)
        return expected
    
    def expected_f(self, f):
        """
        Estimates the expected value of a function with respect to the target distribution.

        Parameters
        ----------
        f : callable
            The function to be applied to each sample.

        Returns
        -------
        tf.Tensor
            The expected value of the function applied to the samples.
        """
        samples = self.samples
        norm = self.normalized_weights[tf.newaxis, :]
        f_values = tf.vectorized_map(f, samples)[:, tf.newaxis]
        expected = tf.matmul(norm, f_values)
        return tf.squeeze(expected)
