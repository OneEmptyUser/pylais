from pylais.utils import buildModelLogp, run_mcmc, flatTensor3D, repeatTensor3D, returnKernel
import tensorflow as tf
import tensorflow_probability as tfp
from pylais.denominators import all_, temporal2, spatial2
from pylais.samples import ISSamples, mcmcSamples
import matplotlib.pyplot as plt

class Lais:
    """Class that implements the LAIS algorithm.
    
    Attributes
    ----------
        loglikelihood : function
            A function provided by the user that calculates the log likelihood of the model.
        logprior : function
            A function provided by the user that calculates the log prior of the model.
        logposterior : function
            A function that calculates the log posterior of the model. This is computed automatically.
        Z : float
            Property that returns the marginal likelihood of the model.
            
    Methods
    -------
        __str__(self)
        main(self, T, N, initial_points, upper_settings={}, lower_settings = {})
            Runs the complete algorithm, from the upper layer to the lower layer.
        upper_layer(self, T, N, initial_points, method="rw", mcmc_settings={})
            Runs the MCMC layer, and adapt the proposals needed in the lower layer with an MCMC algorithm.
        set_means(self, means)
            Allows to skip the MCMC layer. The user can set the MCMC chains as if they were from the upper layer.
        lower_layer(self, cov, M, den="all")
            Runs the IS layer.
        resample(self, n)
            Resamples the samples based on the given number of samples. Calls the method in the ISSamples class.
    """
    def __init__(self, loglikelihood, logprior=None) -> None:
        """
        Initializes a new instance of the class.
        
        Initializes the instance variables `loglikelihood`, `logprior`, and `logposterior` by calling the `buildModelLogp`
        function with the given `loglikelihood` and `logprior` arguments.

        Parameters
        ----------
        loglikelihood : function
            A function that calculates the log likelihood of the model.
        logprior : function, (optional)
            A function that calculates the log prior of the model. Defaults to None.

        Returns
        -------
            None
        """
        self.loglikelihood = loglikelihood
        self.logprior = logprior
        self.logposterior = buildModelLogp(loglikelihood, logprior)
    
    def __str__(self):
        msg = "Lais class\n loglikelihood: {}\n logprior: {}\n".format(self.loglikelihood, self.logprior)
        return msg
    
    def main(self, T, N, initial_points, upper_settings={}, lower_settings = {}):
        """
        Runs the complete algorithm, from the upper layer to the lower layer.
        Calls the upper layer and then the lower layer.

        Parameters
        ----------
        T : int
            The number of iterations for the upper layer.
        N : int
            The number of MCMC chains to run in the upper layer.
        initial_points : tensorflow.Tensor
            The initial points for the MCMC chains.
        upper_settings : dict, optional, default={}
            Additional settings for the upper layer
            The possible keys are:
            
            - "method": The MCMC method to use in the upper layer. Defaults to "rwmh" (Random Walk Metropolis-Hastings).
            - "mcmc_settings": A dictionary of additional settings for the MCMC method. Defaults to an empty dictionary.
            - "targets": A list of functions representing the invariant distributions of the MCMC chains. Defaults to an empty list.
        lower_settings : dict, optional, default={}
            Additional settings for the lower layer.
            The possible keys are
            
            - "cov": The covariance matrix of the proposal distribution. Defaults to the identity matrix.
            - "den": The denominator of the proposal distribution. Defaults to "all".
            - "M": The number of proposal samples. Defaults to 1.
            - "proposal_type": The type of proposal distribution. Defaults to "gaussian". Other value can be "student".
            - "df": The degrees of freedom of the proposal distribution. Defaults to None. Only used if "proposal_type" is "student".

        Returns
        -------
        ImpSamples
            The importance samples generated by the lower layer.
        """
        
        if initial_points.dtype not in [tf.float32, tf.float64]:
            initial_points = tf.cast(initial_points, tf.float64)
        dType = initial_points.dtype
        method = upper_settings.get("method", "rwmh")
        mcmc_settings = upper_settings.get("mcmc_settings", {})
        targets = upper_settings.get("targets", [])
        print("Running MCMC layer.")
        self.upper_layer(T, N, initial_points, method, mcmc_settings, targets)
        
        dim = initial_points.shape[1]
        proposal_cov = lower_settings.get("cov", tf.eye(dim, dtype=dType))
        den = lower_settings.get("den", "all")
        M = lower_settings.get("M", 1)
        proposal_type = lower_settings.get("proposal_type", "gaussian")
        df = lower_settings.get("df", None)
        print("Running IS layer.")
        ImpSamples = self.lower_layer(proposal_cov, M, den, proposal_type, df)
        return ImpSamples
    
    
    def upper_layer(self, T, N, initial_points, method="rwmh", mcmc_settings={}, targets=[]):
        """
    Executes the upper layer of the LAIS algorithm.

    This method runs the MCMC (Markov Chain Monte Carlo) layer of the algorithm, which adapts the proposals needed 
    in the lower layer using an MCMC algorithm. The adaptation is performed by adjusting the MCMC chains to the 
    target distribution. If the `targets` parameter is not empty, each MCMC chain will use the corresponding 
    function in `targets` as its invariant distribution. If `targets` is empty, all chains will use the posterior 
    distribution as their invariant distribution.

    Parameters
    ----------
    T : int
        The number of iterations for the upper layer MCMC algorithm.
    N : int
        The number of MCMC chains to run in the upper layer.
    initial_points : tensorflow.Tensor, shape (N, dim)
        The initial points for the MCMC chains, where `N` is the number of chains, and `dim` is the dimensionality 
        of the parameter space.
    method : str, optional, default="rwmh"
        The MCMC method to use in the upper layer. Defaults to "rwmh" (Random Walk Metropolis-Hastings).
    mcmc_settings : dict, optional, default=None
        A dictionary of additional settings for the MCMC method. If not provided, an empty dictionary is used.
    targets : list of callables, optional, default=None
        A list of functions representing the invariant distributions of the MCMC chains. If provided, the length 
        of `targets` must be equal to `N`. If `targets` is None, all chains will use the posterior distribution.

    Returns
    -------
    mcmcSamples
        An instance of the `mcmcSamples` class containing the MCMC samples obtained from the upper layer.

    Raises
    ------
    ValueError
        If the number of initial points does not equal `N`.
    TypeError
        If `targets` is not a list of functions.
    ValueError
        If `targets` is not empty and its length does not equal `N`.

    Notes
    -----
    The method can utilize various MCMC algorithms, specified via the `method` parameter. The default method is 
    Random Walk Metropolis-Hastings (`rwmh`). The `mcmc_settings` dictionary can be used to fine-tune the MCMC 
    algorithm, such as setting the covariance matrix for proposals. The methods available are:

    - `rwmh`: Random Walk Metropolis-Hastings.
    - `hmc`: Hamiltonian Monte Carlo.
    - `nuts`: No-U-Turn Sampler.
    - `mala`: Metropolis Adjusted Langevin Algorithm.
    - `slice`: Slice Sampling.	

    See the documentation for each method for more details.
    
    Examples
    --------
    .. code-block:: python

        # Example usage of upper_layer
        my_lais = Lais(loglikelihood, logprior)
        T, N, dim = 1000, 3, 1
        initial_points = tf.random.normal(shape=(N, dim))
        mcmc_samples = my_lais.upper_layer(T=T, N=N, initial_points=initial_points)
    """
        
        if initial_points.dtype not in [tf.float32, tf.float64]:
            initial_points = tf.cast(initial_points, tf.float64)
        
        if mcmc_settings.get("cov") is not None:
            mcmc_settings["cov"] = tf.cast(mcmc_settings["cov"], initial_points.dtype)
           
        # get dimensions of the problem
        _, dim = initial_points.shape
        try:
            assert(N ==initial_points.shape[0])
        except:
            raise Exception("N must be equal to the number of initial points")
        
        # kernel = tfp.mcmc.RandomWalkMetropolis(
        #     target_log_prob_fn=self.logposterior
        # )
        # kernel = returnKernel(method, self.logposterior, mcmc_settings)
        
        if targets:
            if not isinstance(targets, list):
                raise("`targets` must be a list of functions.")
            
            print("Using the functions in `targets` as invariant distributions of the MCMC chains.")
            
            if len(targets) != N:
                raise("When using `targets`, `N` must be equal to the number of targets.")
            kernels = [
                returnKernel(method, targets[ikernel], mcmc_settings) for ikernel in range(len(targets))
            ]
        else:
            kernels = N*[returnKernel(method, self.logposterior, mcmc_settings)]
        means = []
        for n in range(N):
            init = initial_points[n]
            aux_means, _ = run_mcmc(kernels[n], T, num_burnin_steps=0,
                                 current_state=init)
            means.append(aux_means)
        
        means = tf.stack(means)
        self.MCMCsamples = mcmcSamples(means)
        return self.MCMCsamples
    
    def set_means(self, means):
        """
        Set the MCMC samples to the given means.

        Parameters
        ----------
        means : tensorflow.Tensor
            The means to set the MCMC samples to.

        Returns
        -------
            None

        Prints:
            str: A message indicating that the means have been set.
            
        Examples
        --------
        .. code-block:: python

            # Example usage of set_means
            lais_instance = Lais(loglikelihood, logprior)
            new_means = tf.constant([[[0.1, 0.2], 
                                     [0.3, 0.4],
                                     [0.5, 0.6]],
                                     [[1, 2], 
                                     [3, 4],
                                     [5, 6]]])
            lais_instance.set_means(new_means)
        """
        self.MCMCsamples = mcmcSamples(means)
        print("Means set.")
    
    def lower_layer(self, cov, M=1, den="all", proposal_type="gaussian", df=None):
        """
        Run the lower layer of the LAIS algorithm.
        
        This function samples from the proposals adapted in the upper layer and assigns the importance weights
        to each sample. It can't be run before the upper layer has been run.

        Parameters
        ----------
        cov : tensorflow.Tensor
            The covariance matrix for the distribution.
        M : int, optional
            The number of samples to be drawn per proposal. Defaults to 1.
        den : str, optional
            The type of denominator to use. Defaults to "all". Options are "all", "spatial" and "temporal".
        proposal_type : str, optional
            The type of proposal distribution. Defaults to "gaussian". Options are "gaussian" and "student".
        df : int, optional
            The degrees of freedom for the Student's t proposal distribution. Defaults to None. Only used if
            proposal_type is "student".

        Returns
        -------
        ISSamples
            The Importance Sampled samples.
            
        Raises
        ------
        ValueError
            If the upper layer has not been run or the means have not been set.

        Prints
        ------
        str
            A message indicating the start and end of the calculation of the denominators.
            
        Examples
        --------
        .. code-block:: python

            # Example usage of lower_layer
            my_lais = Lais(loglikelihood, logprior)
            cov = tf.constant([[1, 0.5], [0.5, 1]])
            my_lais.upper_layer(T=100, N=5, initial_points=tf.zeros((5, 2)))
            my_lais.lower_layer(cov, M=10)
            
        .. code-block:: python

            # Example usage of lower_layer with Student-t proposal
            my_lais = Lais(loglikelihood, logprior)
            cov = tf.constant([[1, 0], [0, 1]], dtype=tf.float64)
            my_lais.upper_layer(T=100, N=5, initial_points=tf.zeros((5, 2)))
            my_lais.lower_layer(cov, M=10, proposal_type="student", df=10)
        """
        # check if upper layer has been run
        if "MCMCsamples" not in self.__dict__:
            raise ValueError("Run the upper layer first or use set_means.")
        
        
        @tf.function(reduce_retracing=True)
        def logposterior(theta):
            return self.logposterior(theta)
        means = self.MCMCsamples.samples
        dType = means.dtype
        cov = tf.cast(cov, dtype=dType) if cov.dtype != dType else cov
        N, T, dim = means.shape
        repeated_means = repeatTensor3D(means, M)
        flatted_repeated_means = flatTensor3D(repeated_means)
        flatted_means = flatTensor3D(means)
        # mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(dim, dtype=tf.float64),
        #                                                          covariance_matrix=cov)
        if proposal_type == "student":
            if df is None:
               df = dim + 1
               
            scale = tf.linalg.cholesky(cov)
            proposal = tfp.distributions.MultivariateStudentTLinearOperator(df,
                                                                            loc=tf.zeros(dim, dtype=dType),
                                                                            scale=tf.linalg.LinearOperatorLowerTriangular(scale))
            
        if proposal_type == "gaussian":
            proposal = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(dim, dtype=dType),
                                                        scale_tril=tf.linalg.cholesky(cov))
        
        
        flatted_samples = proposal.sample(M*T*N) + flatted_repeated_means
        samples = tf.reshape(flatted_samples, (N, T*M, dim))
        
        # calculate the denominators
        print("Calculating weights: numerator")
        numerator = tf.map_fn(fn=logposterior, elems=flatted_samples)
        numerator = tf.exp(numerator)
        print("Calculating weights: denominator")
        proposal_settings = {"proposal_type": proposal_type, "df": df, "cov": cov}
        if den == "temporal":
            # denominator = temporal(means, samples, proposal_settings)
            denominator = temporal2(means, flatted_samples, proposal_settings)
        elif den == "spatial":
            # denominator = spatial(means, samples, proposal_settings)
            denominator = spatial2(means, flatted_samples, proposal_settings)
        else:
            denominator  = all_(flatted_means, flatted_samples, proposal_settings)
        print("Calculating weights: done")
        weights = numerator/denominator
        
        self.samples = samples
        # self.flatted_samples = flatted_samples
        self.weights = weights
        
        self.IS_samples = ISSamples(flatted_samples, weights)
        return self.IS_samples
        
    def resample(self, n):
        """
        Resamples the samples based on the given number of samples.

        This function calls the homonymous resample method of the ISSamples class.
        
        Parameters
        ----------
        n : int
            The number of samples to resample.

        Returns
        -------
        Union[tensorflow.Tensor, str]
            If the IS_samples attribute is present, returns the resampled samples.
            Otherwise, returns the string "No IS samples".
        """        
        if "IS_samples" in dir(self):
            return self.IS_samples.resample(n)
        else:
            return "No IS samples"
        
        # norm_weights = self.weights/tf.math.reduce_sum(self.weights)
        # idx = tf.random.categorical(tf.math.log([norm_weights]), n)
        # return tf.gather(self.flatted_samples, tf.squeeze(idx))
        
    @property
    def Z(self):
        """
        Returns the mean value of the weights if the IS_samples attribute is present,
        otherwise returns the string "No IS samples".

        Returns:
            Union[float, str]: The mean value of the weights if the IS_samples attribute is present,
            otherwise returns the string "No IS samples".
        """
        if "IS_samples" in dir(self):
            return self.IS_samples.Z
        else:
            return "No IS samples"
        
        
    def histogram(self, n_samples, dimensions=(0,1), bins=25, axis=None, **hist_args):
        """
        Generate a histogram plot of the IS samples.
        
        Resample `n_samples` from the IS samples and plot the histogram of the selected dimensions.
        The `dimensions` argument specify the columns to plot in the histogram. To get the histogram
        of only one component `n` use `(n,)`.

        Parameters
        ----------
        n_samples : int
            The number of samples to resample.
        dimensions : tuple, optional
            The dimensions of the samples to plot. Defaults to (0, 1). To get the histogram
            of only one component `n` use `(n,)`.
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            The axis on which to plot the histogram. If not provided, a new axis will be created. Defaults to None.
        **hist_args
            Additional keyword arguments to pass to the histogram function.

        Returns
        -------
        None
        """
        
        if "IS_samples" not in self.__dict__:
            raise("No IS samples.")

        if axis:
            ax = axis
        else:
            fig, ax = plt.subplots()

        samples = self.resample(n_samples)
        if len(dimensions) == 2:
            hist = ax.hist2d(samples.samples[:, dimensions[0]], samples.samples[:, dimensions[1]],
                      bins=int(bins),
                      cmap='Blues',
                      density=True,
                      **hist_args)
            plt.colorbar(hist[3], ax=ax, shrink=0.8)
            ax.set_xlabel(r"$\theta_{}$".format(dimensions[0]))
            ax.set_ylabel(r"$\theta_{}$".format(dimensions[1]))
        elif len(dimensions) == 1:
            ax.hist(samples.samples[:, dimensions[0]],
                    bins=int(bins),
                    density=True,
                    **hist_args
                    )
            ax.set_xlabel(r"$\theta_{}$".format(dimensions[0]))
        ax.set_title(f"IS histogram of {n_samples} resampled samples")
        plt.show()