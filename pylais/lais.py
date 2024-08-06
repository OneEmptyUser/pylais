from pylais.utils import buildModelLogp, run_mcmc, flatTensor3D, repeatTensor3D, returnKernel
import tensorflow as tf
import tensorflow_probability as tfp
from pylais.denominators import all_, temporal, spatial
from pylais.samples import ISSamples, mcmcSamples
import matplotlib.pyplot as plt

class Lais:
    """Class that implements the LAIS algorithm.
    
    Attributes
    ----------
        loglikelihood : function
            A function that calculates the log likelihood of the model.
        logprior : function
            A function that calculates the log prior of the model.
        logposterior : function
            A function that calculates the log posterior of the model.
        Z : float
            The marginal likelihood of the model.
            
    Methods
    -------
        __str__(self)
        main(self, n_iter, N, initial_points, upper_settings={}, lower_settings = {})
            Runs the complete algorithm, from the upper layer to the lower layer.
        upper_layer(self, n_iter, N, initial_points, method="rw", mcmc_settings={})
            Runs the MCMC layer, and adapt the proposals needed in the lower layer with an MCMC algorithm.
        set_means(self, means)
            Allows to skip the MCMC layer. The user can the MCMC chains as if they were from the upper layer.
        lower_layer(self, cov, n_per_sample, den="all")
            Runs the IS layer.
        resample(self, n)
            Resamples the samples based on the given number of samples. Calls the method in the ISSamples class.
    """
    def __init__(self, loglikelihood, logprior=None) -> None:
        """
        Initializes a new instance of the class.

        Args:
            loglikelihood: function
                A function that calculates the log likelihood of the model.
            logprior: function, (optional)
                A function that calculates the log prior of the model. Defaults to None.

        Returns:
            None

        Initializes the instance variables `loglikelihood`, `logprior`, and `logposterior` by calling the `buildModelLogp` function with the given `loglikelihood` and `logprior` arguments.
        """
        self.loglikelihood = loglikelihood
        self.logprior = logprior
        self.logposterior = buildModelLogp(loglikelihood, logprior)
    
    def __str__(self):
        msg = "Lais class\n loglikelihood: {}\n logprior: {}\n".format(self.loglikelihood, self.logprior)
        return msg
    
    def main(self, n_iter, N, initial_points, upper_settings={}, lower_settings = {}):
        """
        Runs the complete algorithm, from the upper layer to the lower layer.

        Args:
            n_iter: int
                The number of iterations for the upper layer.
            N: int
                The number of MCMC chains to run in the upper layer.
            initial_points: tensorflow.Tensor
                The initial points for the MCMC chains.
            upper_settings: dict, (optional)
                Additional settings for the upper layer. Defaults to an empty dictionary.
                - method (str, optional): The method to use in the upper layer. Defaults to "mcmc".
                - mcmc_settings (dict, optional): Additional settings for the MCMC method. Defaults to an empty dictionary.
            lower_settings: dict, (optional)
                Additional settings for the lower layer. Defaults to an empty dictionary.
                - cov (tf.Tensor, optional): The covariance matrix for the proposal distribution. Defaults to the identity matrix.
                - den (str, optional): The type of importance sampling to use. Defaults to "all".
                - n_per_sample (int, optional): The number of samples to draw for each importance sampling step. Defaults to 1.

        Returns:
            ISSamples: The importance sampling samples.

        """
        method = upper_settings.get("method", "mcmc")
        mcmc_settings = upper_settings.get("mcmc_settings", {})
        print("Running MCMC layer.")
        self.upper_layer(n_iter, N, initial_points, method, mcmc_settings)
        dim = initial_points.shape[1]
        proposal_cov = lower_settings.get("cov", tf.eye(dim, dtype=tf.float64))
        den = lower_settings.get("den", "all")
        n_per_sample = lower_settings.get("n_per_sample", 1)
        print("Running IS layer.")
        ImpSamples = self.lower_layer(proposal_cov, n_per_sample, den)
        return ImpSamples
    
    
    def upper_layer(self, n_iter, N, initial_points, method="mcmc", mcmc_settings={}, targets=[]):
        """
        Run the upper layer of the algorithm.
        
        This function runs the MCMC layer, and adapt the proposals needed in the lower layer with an MCMC algorithm,
        which consists of adapting the MCMC chains to the target distribution. If `targets` is not empty, the MCMC chains
        take as invariant density the function in `targets` that take that index, i.e, chain `n` has as invariant density
        the function `targets[n]`. If it is empty, all chains will have as invariant density the posterior.

        Parameters
        ----------
        n_iter : int
            The number of iterations for the upper layer.
        N : int
            The number of MCMC chains to run in the upper layer.
        initial_points : tensorflow.Tensor
            The initial points for the MCMC chains. It is a tf.Tensor of shape (N, dim).
        method : str, optional
            The method to use in the upper layer. Defaults to "mcmc".
        mcmc_settings : dict, optional
            Additional settings for the MCMC method. Defaults to an empty dictionary.
        targets : list, optional
            The functions to use as invariant distributions of the MCMC chains. Defaults to an empty list.

        Returns
        -------
        mcmcSamples
            The MCMC samples obtained from the upper layer.

        Raises
        ------
        Exception
            If the number of initial points is not equal to N.
        Exception
            If `targets` is not a list of functions.
        Exception
            If `targets` is not empty and its length is not equal to N.
        """
        
        # get dimensions of the problem
        _, dim = initial_points.shape
        try:
            assert(N ==initial_points.shape[0])
        except:
            raise Exception("N must be equal to the number of initial points")
        
        # kernel = tfp.mcmc.RandomWalkMetropolis(
        #     target_log_prob_fn=self.logposterior
        # )
        kernel = returnKernel(method, self.logposterior, mcmc_settings)
        
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
            aux_means, _ = run_mcmc(kernels[n], n_iter, num_burnin_steps=0,
                                 current_state=init)
            means.append(aux_means)
        
        means = tf.stack(means)
        self.MCMCsamples = mcmcSamples(means)
        return self.MCMCsamples
    
    def set_means(self, means):
        """
        Set the MCMC samples to the given means.

        Parameters:
            means: tensorflow.Tensor
                The means to set the MCMC samples to.

        Returns:
            None

        Prints:
            str: A message indicating that the means have been set.
        """
        self.MCMCsamples = mcmcSamples(means)
        print("Means set.")
    
    def lower_layer(self, cov, n_per_sample, den="all", proposal_type="gaussian", df=None):
        """
        Run the lower layer of the LAIS algorithm.
        
        This function samples from the proposals adapted in the upper layer and assigns the importance weights
        to each sample.

        Parameters
        ----------
        cov : tensorflow.Tensor
            The covariance matrix for the distribution.
        n_per_sample : int
            The number of samples per iteration.
        den : str, optional
            The type of denominator to use. Defaults to "all".

        Returns
        -------
        ISSamples
            The importance sampling samples.

        Raises
        ------
        ValueError
            If the upper layer has not been run or the means have not been set.

        Prints
        ------
        str
            A message indicating the start and end of the calculation of the denominators.
        """
        # check if upper layer has been run
        if "MCMCsamples" not in self.__dict__:
            raise ValueError("Run the upper layer first or use set_means.")
        
        
        @tf.function
        def logposterior(theta):
            return self.logposterior(theta)
        means = self.MCMCsamples.samples
        N, n_iter, dim = means.shape
        repeated_means = repeatTensor3D(means, n_per_sample)
        flatted_repeated_means = flatTensor3D(repeated_means)
        flatted_means = flatTensor3D(means)
        # mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(dim, dtype=tf.float64),
        #                                                          covariance_matrix=cov)
        if proposal_type == "student":
            if df is None:
               df = dim + 1
               
            scale = tf.linalg.cholesky(cov)
            proposal = tfp.distributions.MultivariateStudentTLinearOperator(df,
                                                                            loc=tf.zeros(dim, dtype=tf.float64),
                                                                            scale=tf.linalg.LinearOperatorLowerTriangular(scale))
            
        if proposal_type == "gaussian":
            proposal = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(dim, dtype=tf.float64),
                                                        scale_tril=tf.linalg.cholesky(cov))
        
        
        flatted_samples = proposal.sample(n_per_sample*n_iter*N) + flatted_repeated_means
        samples = tf.reshape(flatted_samples, (N,n_iter*n_per_sample, dim))
        
        # calculate the denominators
        print("Calculating weights: numerator")
        numerator = tf.map_fn(fn=logposterior, elems=flatted_samples)
        numerator = tf.exp(numerator)
        print("Calculating weights: denominator")
        proposal_settings = {"proposal_type": proposal_type, "df": df, "cov": cov}
        if den == "temporal":
            denominator = temporal(means, samples, proposal_settings)
        elif den == "spatial":
            denominator = spatial(means, samples, proposal_settings)
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
        
        
    def histogram(self, n_samples, dimensions=(0,1), axis=None, **hist_args):
        """
        Generate a histogram plot of the IS samples.

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
                      bins=int(tf.math.sqrt(tf.cast(n_samples, dtype=tf.float32)).numpy()),
                      cmap='Blues',
                      density=True,
                      **hist_args)
            plt.colorbar(hist[3], ax=ax, shrink=0.8)
            ax.set_xlabel(r"$\theta_{}$".format(dimensions[0]))
            ax.set_ylabel(r"$\theta_{}$".format(dimensions[1]))
        elif len(dimensions) == 1:
            ax.hist(samples.samples[:, dimensions[0]],
                    bins=int(tf.math.sqrt(tf.cast(n_samples, dtype=tf.float32)).numpy()),
                    density=True,
                    **hist_args
                    )
            ax.set_xlabel(r"$\theta_{}$".format(dimensions[0]))
        ax.set_title(f"IS histogram of {n_samples} resampled samples")
        plt.show()