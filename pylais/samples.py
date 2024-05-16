import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import corrcoef
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
# from tfp.stats import correlation

class mcmcSamples:
    def __init__(self, samples):
        self.samples = samples

    def __str__(self):
        return f"{len(self.samples)} MCMC chains.\nGelman-Rubin statistic: {self.gelman_rubin()}"

    def gelman_rubin(self):
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
        
    def trace(self, axis=None):
        if axis:
            ax = axis
        else:
            _, ax = plt.subplots()
    
        n_chains = self.samples.shape[0]
        for n in range(n_chains):
            ax.plot(self.samples[n, :, :])
            
        plt.show()
        
    def autoCorrPlot(self,chain=0, component=0, max_lag=10, axis=None, plot_args={}):
        
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
        plt.show()
        
    
    def cumulativeMean(self):
        dType = self.samples.dtype
        n_chains, iterations, _ = self.samples.shape
        for n in range(n_chains):
            plt.plot(tf.math.cumsum(self.samples[n, :, :], axis=0)/tf.range(1, iterations+1,
                                                                            dtype=dType)[:, tf.newaxis])
        plt.show()
    
    def scatter(self, xlim=None, ylim=None, axis=None):
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
    
class Samples:
    def __init__(self, samples, weights):
        self.samples = samples
        self.weights = weights
        self.normalized_weights = weights / tf.math.reduce_sum(weights)
    def resample(self, n):
        norm_weights = self.normalized_weights
        idx = tf.random.categorical(tf.math.log([norm_weights]), n)
        return tf.gather(self.samples, tf.squeeze(idx))
    
    @property
    def ess(self):
        return 1 / tf.math.reduce_sum(tf.math.square(self.normalized_weights))
    @property
    def Z(self):
        return tf.math.reduce_mean(self.weights)
    
    def moment_n(self, n=1):
        samples = self.samples
        # flatted_normalized_weights = self.normalized_weights
        norm = self.normalized_weights[tf.newaxis, :]
        expected = tf.matmul(norm, samples**n)
        return expected
    
    def expected_f(self, f):
        samples = self.samples
        norm = self.normalized_weights[tf.newaxis, :]
        f_values = tf.vectorized_map(f, samples)[:, tf.newaxis]
        expected = tf.matmul(norm, f_values)
        return tf.squeeze(expected)
    
    def __str__(self):
        return f"{len(self.weights)} Samples class with ESS = {self.ess} and Z = {self.Z}"