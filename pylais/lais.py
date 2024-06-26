from .utils import buildModelLogp, run_mcmc, flatTensor3D, repeatTensor3D, returnKernel
import tensorflow as tf
import tensorflow_probability as tfp
from .denominators import all_, temporal, spatial
from.samples import ISSamples, mcmcSamples


class Lais:
    def __init__(self, loglikelihood, logprior=None) -> None:
        self.loglikelihood = loglikelihood
        self.logprior = logprior
        self.logposterior = buildModelLogp(loglikelihood, logprior)
    
    def __str__(self):
        msg = "Lais class\n loglikelihood: {}\n logprior: {}\n".format(self.loglikelihood, self.logprior)
        return msg
    
    def main(self, n_iter, N, initial_points, upper_settings={}, lower_settings = {}):
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
    
    
    def upper_layer(self, n_iter, N, initial_points, method="rw", mcmc_settings={}):
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
        means = []
        for n in range(N):
            init = initial_points[n]
            aux_means, _ = run_mcmc(kernel, n_iter, num_burnin_steps=0,
                                 current_state=init)
            means.append(aux_means)
        
        means = tf.stack(means)
        self.MCMCsamples = mcmcSamples(means)
        return self.MCMCsamples
    
    def set_means(self, means):
        self.MCMCsamples = mcmcSamples(means)
        print("Means set.")
    
    def lower_layer(self, cov, n_per_sample, den="all"):
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
        mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(dim, dtype=tf.float64),
                                                                 covariance_matrix=cov)
        
        flatted_samples = mvn.sample(n_per_sample*n_iter*N) + flatted_repeated_means
        samples = tf.reshape(flatted_samples, (N,n_iter*n_per_sample, dim))
        
        # calculate the denominators
        print("Calculating denominators: numerator")
        numerator = tf.map_fn(fn=logposterior, elems=flatted_samples)
        numerator = tf.exp(numerator)
        print("Calculating denominators: denominator")
        if den == "temporal":
            denominator = temporal(means, samples, cov)
        elif den == "spatial":
            denominator = spatial(means, samples, cov)
        else:
            denominator  = all_(flatted_repeated_means, flatted_samples, cov)
        print("Calculating denominators: done")
        weights = numerator/denominator
        
        self.samples = samples
        # self.flatted_samples = flatted_samples
        self.weights = weights
        
        self.IS_samples = ISSamples(flatted_samples, weights)
        return self.IS_samples
        
    def resample(self, n):
        
        if "IS_samples" in dir(self):
            return self.IS_samples.resample(n)
        else:
            return "No IS samples"
        
        # norm_weights = self.weights/tf.math.reduce_sum(self.weights)
        # idx = tf.random.categorical(tf.math.log([norm_weights]), n)
        # return tf.gather(self.flatted_samples, tf.squeeze(idx))
        
    @property
    def Z(self):
        if "IS_samples" in dir(self):
            return self.IS_samples.Z
        else:
            return "No IS samples"