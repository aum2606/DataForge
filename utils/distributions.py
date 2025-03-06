"""
Functions for generating data with various statistical distributions.
"""

import numpy as np
from scipy import stats

def generate_uniform(size=1, low=0, high=1):
    """
    Generate data with a uniform distribution.
    
    Args:
        size (int): Number of samples to generate
        low (float): Lower bound
        high (float): Upper bound
        
    Returns:
        numpy.ndarray: Array of uniformly distributed random numbers
    """
    return np.random.uniform(low, high, size)

def generate_normal(size=1, mean=0, std=1):
    """
    Generate data with a normal (Gaussian) distribution.
    
    Args:
        size (int): Number of samples to generate
        mean (float): Mean of the distribution
        std (float): Standard deviation of the distribution
        
    Returns:
        numpy.ndarray: Array of normally distributed random numbers
    """
    return np.random.normal(mean, std, size)

def generate_lognormal(size=1, mean=0, sigma=1):
    """
    Generate data with a log-normal distribution.
    
    Args:
        size (int): Number of samples to generate
        mean (float): Mean of the underlying normal distribution
        sigma (float): Standard deviation of the underlying normal distribution
        
    Returns:
        numpy.ndarray: Array of log-normally distributed random numbers
    """
    return np.random.lognormal(mean, sigma, size)

def generate_exponential(size=1, scale=1.0):
    """
    Generate data with an exponential distribution.
    
    Args:
        size (int): Number of samples to generate
        scale (float): Scale parameter (inverse of rate parameter)
        
    Returns:
        numpy.ndarray: Array of exponentially distributed random numbers
    """
    return np.random.exponential(scale, size)

def generate_poisson(size=1, lam=1.0):
    """
    Generate data with a Poisson distribution.
    
    Args:
        size (int): Number of samples to generate
        lam (float): Rate parameter
        
    Returns:
        numpy.ndarray: Array of Poisson distributed random numbers
    """
    return np.random.poisson(lam, size)

def generate_binomial(size=1, n=1, p=0.5):
    """
    Generate data with a binomial distribution.
    
    Args:
        size (int): Number of samples to generate
        n (int): Number of trials
        p (float): Probability of success in each trial
        
    Returns:
        numpy.ndarray: Array of binomially distributed random numbers
    """
    return np.random.binomial(n, p, size)

def generate_beta(size=1, a=1, b=1):
    """
    Generate data with a beta distribution.
    
    Args:
        size (int): Number of samples to generate
        a (float): Alpha parameter
        b (float): Beta parameter
        
    Returns:
        numpy.ndarray: Array of beta distributed random numbers
    """
    return np.random.beta(a, b, size)

def generate_gamma(size=1, shape=1, scale=1):
    """
    Generate data with a gamma distribution.
    
    Args:
        size (int): Number of samples to generate
        shape (float): Shape parameter
        scale (float): Scale parameter
        
    Returns:
        numpy.ndarray: Array of gamma distributed random numbers
    """
    return np.random.gamma(shape, scale, size)

def generate_weibull(size=1, a=1):
    """
    Generate data with a Weibull distribution.
    
    Args:
        size (int): Number of samples to generate
        a (float): Shape parameter
        
    Returns:
        numpy.ndarray: Array of Weibull distributed random numbers
    """
    return np.random.weibull(a, size)

def generate_pareto(size=1, a=1):
    """
    Generate data with a Pareto distribution.
    
    Args:
        size (int): Number of samples to generate
        a (float): Shape parameter
        
    Returns:
        numpy.ndarray: Array of Pareto distributed random numbers
    """
    return np.random.pareto(a, size)

def generate_multivariate_normal(size=1, mean=None, cov=None):
    """
    Generate data with a multivariate normal distribution.
    
    Args:
        size (int): Number of samples to generate
        mean (array_like): Mean of the distribution (1-D array)
        cov (array_like): Covariance matrix (2-D array)
        
    Returns:
        numpy.ndarray: Array of multivariate normally distributed random numbers
    """
    if mean is None:
        mean = [0, 0]
    if cov is None:
        cov = [[1, 0], [0, 1]]
    
    return np.random.multivariate_normal(mean, cov, size)

def generate_mixture(size=1, distributions=None, weights=None):
    """
    Generate data from a mixture of distributions.
    
    Args:
        size (int): Number of samples to generate
        distributions (list): List of distribution functions
        weights (list): List of weights for each distribution
        
    Returns:
        numpy.ndarray: Array of random numbers from the mixture distribution
    """
    if distributions is None:
        distributions = [generate_normal, generate_normal]
    if weights is None:
        weights = [0.5, 0.5]
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Generate indices based on weights
    indices = np.random.choice(len(distributions), size=size, p=weights)
    
    # Generate samples from each distribution
    samples = np.zeros(size)
    for i, dist in enumerate(distributions):
        mask = (indices == i)
        if np.any(mask):
            samples[mask] = dist(size=np.sum(mask))
    
    return samples

def generate_time_series_with_trend(size=100, trend_coefficient=0.1, noise_level=1.0, start_value=0):
    """
    Generate a time series with a linear trend.
    
    Args:
        size (int): Number of time points
        trend_coefficient (float): Slope of the trend
        noise_level (float): Standard deviation of the noise
        start_value (float): Starting value of the time series
        
    Returns:
        numpy.ndarray: Time series data
    """
    time = np.arange(size)
    trend = start_value + trend_coefficient * time
    noise = np.random.normal(0, noise_level, size)
    return trend + noise

def generate_time_series_with_seasonality(size=100, period=12, amplitude=1.0, noise_level=0.5):
    """
    Generate a time series with seasonality.
    
    Args:
        size (int): Number of time points
        period (int): Period of the seasonality
        amplitude (float): Amplitude of the seasonal component
        noise_level (float): Standard deviation of the noise
        
    Returns:
        numpy.ndarray: Time series data
    """
    time = np.arange(size)
    seasonality = amplitude * np.sin(2 * np.pi * time / period)
    noise = np.random.normal(0, noise_level, size)
    return seasonality + noise
