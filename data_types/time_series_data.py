"""
Module for generating synthetic time series data.
"""

import numpy as np
import pandas as pd
import random
import datetime
import os
import sys
from typing import List, Dict, Tuple, Any, Optional, Union

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules
from utils import distributions
from config import config

def generate_time_index(start_date: str = '2020-01-01', periods: int = 100, 
                       freq: str = 'D', include_weekends: bool = True) -> pd.DatetimeIndex:
    """
    Generate a time index for time series data.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        periods (int): Number of periods to generate
        freq (str): Frequency of the time index ('D' for daily, 'H' for hourly, etc.)
        include_weekends (bool): Whether to include weekends
        
    Returns:
        pandas.DatetimeIndex: Generated time index
    """
    # Generate the time index
    time_index = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Filter out weekends if requested
    if not include_weekends:
        time_index = time_index[time_index.dayofweek < 5]  # 0-4 are weekdays
    
    return time_index

def generate_trend(time_index: pd.DatetimeIndex, trend_type: str = 'linear', 
                  coefficient: float = 0.1, start_value: float = 0) -> np.ndarray:
    """
    Generate a trend component for time series data.
    
    Args:
        time_index (pandas.DatetimeIndex): Time index
        trend_type (str): Type of trend ('linear', 'quadratic', 'exponential', 'logarithmic')
        coefficient (float): Coefficient for the trend
        start_value (float): Starting value for the trend
        
    Returns:
        numpy.ndarray: Trend component
    """
    # Convert time index to numeric values (days since start)
    t = np.arange(len(time_index))
    
    if trend_type == 'linear':
        # Linear trend: y = a*t + b
        trend = start_value + coefficient * t
    
    elif trend_type == 'quadratic':
        # Quadratic trend: y = a*t^2 + b
        trend = start_value + coefficient * t**2
    
    elif trend_type == 'exponential':
        # Exponential trend: y = a*e^(b*t)
        trend = start_value + np.exp(coefficient * t)
    
    elif trend_type == 'logarithmic':
        # Logarithmic trend: y = a*log(t+1) + b
        trend = start_value + coefficient * np.log1p(t)
    
    else:
        # Default to linear trend
        trend = start_value + coefficient * t
    
    return trend

def generate_seasonality(time_index: pd.DatetimeIndex, period: int = 12, 
                        amplitude: float = 1.0, phase_shift: float = 0) -> np.ndarray:
    """
    Generate a seasonality component for time series data.
    
    Args:
        time_index (pandas.DatetimeIndex): Time index
        period (int): Period of the seasonality
        amplitude (float): Amplitude of the seasonal component
        phase_shift (float): Phase shift of the seasonal component
        
    Returns:
        numpy.ndarray: Seasonality component
    """
    # Convert time index to numeric values (days since start)
    t = np.arange(len(time_index))
    
    # Generate seasonality using sine function
    seasonality = amplitude * np.sin(2 * np.pi * (t + phase_shift) / period)
    
    return seasonality

def generate_cyclical_pattern(time_index: pd.DatetimeIndex, pattern: List[float] = None) -> np.ndarray:
    """
    Generate a cyclical pattern component for time series data.
    
    Args:
        time_index (pandas.DatetimeIndex): Time index
        pattern (list, optional): List of values representing the pattern
        
    Returns:
        numpy.ndarray: Cyclical pattern component
    """
    # Default pattern if not provided
    if pattern is None:
        pattern = [0, 1, 2, 3, 2, 1]
    
    # Repeat the pattern to match the length of the time index
    n_repeats = len(time_index) // len(pattern) + 1
    repeated_pattern = np.tile(pattern, n_repeats)
    
    # Truncate to match the length of the time index
    cyclical = repeated_pattern[:len(time_index)]
    
    return cyclical

def generate_noise(time_index: pd.DatetimeIndex, noise_type: str = 'gaussian', 
                  noise_level: float = 0.5, **kwargs) -> np.ndarray:
    """
    Generate a noise component for time series data.
    
    Args:
        time_index (pandas.DatetimeIndex): Time index
        noise_type (str): Type of noise ('gaussian', 'uniform', 'autoregressive')
        noise_level (float): Level of noise
        **kwargs: Additional arguments for specific noise types
        
    Returns:
        numpy.ndarray: Noise component
    """
    size = len(time_index)
    
    if noise_type == 'gaussian':
        # Gaussian noise
        noise = np.random.normal(0, noise_level, size)
    
    elif noise_type == 'uniform':
        # Uniform noise
        noise = np.random.uniform(-noise_level, noise_level, size)
    
    elif noise_type == 'autoregressive':
        # Autoregressive noise (AR(1) process)
        ar_coef = kwargs.get('ar_coefficient', 0.7)
        noise = np.zeros(size)
        noise[0] = np.random.normal(0, noise_level)
        
        for i in range(1, size):
            noise[i] = ar_coef * noise[i-1] + np.random.normal(0, noise_level)
    
    else:
        # Default to Gaussian noise
        noise = np.random.normal(0, noise_level, size)
    
    return noise

def generate_anomalies(time_index: pd.DatetimeIndex, base_series: np.ndarray, 
                      anomaly_ratio: float = 0.05, anomaly_type: str = 'point',
                      min_anomaly_value: float = None, max_anomaly_value: float = None) -> Tuple[np.ndarray, List[int]]:
    """
    Generate anomalies in a time series.
    
    Args:
        time_index (pandas.DatetimeIndex): Time index
        base_series (numpy.ndarray): Base time series data
        anomaly_ratio (float): Ratio of anomalies to introduce
        anomaly_type (str): Type of anomalies ('point', 'collective', 'contextual')
        min_anomaly_value (float, optional): Minimum value for anomalies
        max_anomaly_value (float, optional): Maximum value for anomalies
        
    Returns:
        tuple: (time_series_with_anomalies, anomaly_indices)
            - time_series_with_anomalies (numpy.ndarray): Time series with anomalies
            - anomaly_indices (list): Indices of anomalies
    """
    size = len(time_index)
    series_with_anomalies = base_series.copy()
    
    # Calculate number of anomalies
    num_anomalies = int(size * anomaly_ratio)
    
    # Set default anomaly value range if not provided
    if min_anomaly_value is None:
        min_anomaly_value = np.min(base_series) - 3 * np.std(base_series)
    if max_anomaly_value is None:
        max_anomaly_value = np.max(base_series) + 3 * np.std(base_series)
    
    if anomaly_type == 'point':
        # Point anomalies: individual points that deviate significantly
        anomaly_indices = random.sample(range(size), num_anomalies)
        
        for idx in anomaly_indices:
            # Generate a random anomaly value
            anomaly_value = random.uniform(min_anomaly_value, max_anomaly_value)
            series_with_anomalies[idx] = anomaly_value
    
    elif anomaly_type == 'collective':
        # Collective anomalies: subsequences that deviate significantly
        # Choose a random starting point for each anomaly sequence
        anomaly_indices = []
        sequence_length = min(5, size // 20)  # Default sequence length
        
        for _ in range(num_anomalies // sequence_length + 1):
            start_idx = random.randint(0, size - sequence_length - 1)
            
            # Generate anomaly values for the sequence
            for i in range(sequence_length):
                idx = start_idx + i
                if idx < size:
                    anomaly_value = random.uniform(min_anomaly_value, max_anomaly_value)
                    series_with_anomalies[idx] = anomaly_value
                    anomaly_indices.append(idx)
    
    elif anomaly_type == 'contextual':
        # Contextual anomalies: points that are anomalous in a specific context
        # For simplicity, we'll consider weekends as a context
        anomaly_indices = []
        
        for i, date in enumerate(time_index):
            # Check if it's a weekend
            if date.dayofweek >= 5:  # 5 and 6 are weekend days
                if random.random() < anomaly_ratio * 2:  # Higher probability on weekends
                    anomaly_value = random.uniform(min_anomaly_value, max_anomaly_value)
                    series_with_anomalies[i] = anomaly_value
                    anomaly_indices.append(i)
    
    else:
        # Default to point anomalies
        anomaly_indices = random.sample(range(size), num_anomalies)
        
        for idx in anomaly_indices:
            # Generate a random anomaly value
            anomaly_value = random.uniform(min_anomaly_value, max_anomaly_value)
            series_with_anomalies[idx] = anomaly_value
    
    return series_with_anomalies, anomaly_indices

def generate_missing_values(time_series: np.ndarray, missing_ratio: float = 0.05, 
                           missing_pattern: str = 'random') -> np.ndarray:
    """
    Generate missing values in a time series.
    
    Args:
        time_series (numpy.ndarray): Time series data
        missing_ratio (float): Ratio of missing values to introduce
        missing_pattern (str): Pattern of missing values ('random', 'consecutive', 'periodic')
        
    Returns:
        numpy.ndarray: Time series with missing values (NaN)
    """
    size = len(time_series)
    series_with_missing = time_series.copy()
    
    # Calculate number of missing values
    num_missing = int(size * missing_ratio)
    
    if missing_pattern == 'random':
        # Random missing values
        missing_indices = random.sample(range(size), num_missing)
        
        for idx in missing_indices:
            series_with_missing[idx] = np.nan
    
    elif missing_pattern == 'consecutive':
        # Consecutive missing values
        # Choose a random starting point
        start_idx = random.randint(0, size - num_missing - 1)
        
        for i in range(num_missing):
            series_with_missing[start_idx + i] = np.nan
    
    elif missing_pattern == 'periodic':
        # Periodic missing values
        period = max(1, size // num_missing)
        
        for i in range(0, size, period):
            if i < size:
                series_with_missing[i] = np.nan
    
    else:
        # Default to random missing values
        missing_indices = random.sample(range(size), num_missing)
        
        for idx in missing_indices:
            series_with_missing[idx] = np.nan
    
    return series_with_missing

def generate_time_series(length: int = None, components: List[str] = None,
                        start_date: str = '2020-01-01', freq: str = 'D',
                        include_weekends: bool = True, **kwargs) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Generate a synthetic time series with specified components.
    
    Args:
        length (int, optional): Length of the time series
        components (list, optional): List of components to include ('trend', 'seasonality', 'cyclical', 'noise')
        start_date (str): Start date in 'YYYY-MM-DD' format
        freq (str): Frequency of the time series ('D' for daily, 'H' for hourly, etc.)
        include_weekends (bool): Whether to include weekends
        **kwargs: Additional arguments for specific components
            - trend_type (str): Type of trend ('linear', 'quadratic', 'exponential', 'logarithmic')
            - trend_coefficient (float): Coefficient for the trend
            - trend_start_value (float): Starting value for the trend
            - seasonality_period (int): Period of the seasonality
            - seasonality_amplitude (float): Amplitude of the seasonal component
            - seasonality_phase_shift (float): Phase shift of the seasonal component
            - cyclical_pattern (list): List of values representing the cyclical pattern
            - noise_type (str): Type of noise ('gaussian', 'uniform', 'autoregressive')
            - noise_level (float): Level of noise
            - ar_coefficient (float): Coefficient for autoregressive noise
            - add_anomalies (bool): Whether to add anomalies
            - anomaly_ratio (float): Ratio of anomalies to introduce
            - anomaly_type (str): Type of anomalies ('point', 'collective', 'contextual')
            - add_missing_values (bool): Whether to add missing values
            - missing_ratio (float): Ratio of missing values to introduce
            - missing_pattern (str): Pattern of missing values ('random', 'consecutive', 'periodic')
        
    Returns:
        tuple: (time_index, time_series)
            - time_index (pandas.DatetimeIndex): Time index
            - time_series (numpy.ndarray): Generated time series data
    """
    # Use default values from config if not specified
    length = length or config.get('time_series.default_length', 100)
    
    if components is None:
        components = ['trend', 'seasonality', 'noise']
    
    # Generate time index
    time_index = generate_time_index(start_date, length, freq, include_weekends)
    
    # Adjust length if weekends are excluded
    actual_length = len(time_index)
    
    # Initialize time series
    time_series = np.zeros(actual_length)
    
    # Add trend component
    if 'trend' in components:
        trend_type = kwargs.get('trend_type', 'linear')
        trend_coefficient = kwargs.get('trend_coefficient', config.get('time_series.default_trend_coefficient', 0.1))
        trend_start_value = kwargs.get('trend_start_value', 0)
        
        trend = generate_trend(time_index, trend_type, trend_coefficient, trend_start_value)
        time_series += trend
    
    # Add seasonality component
    if 'seasonality' in components:
        seasonality_period = kwargs.get('seasonality_period', config.get('time_series.default_seasonality_period', 12))
        seasonality_amplitude = kwargs.get('seasonality_amplitude', 1.0)
        seasonality_phase_shift = kwargs.get('seasonality_phase_shift', 0)
        
        seasonality = generate_seasonality(time_index, seasonality_period, seasonality_amplitude, seasonality_phase_shift)
        time_series += seasonality
    
    # Add cyclical component
    if 'cyclical' in components:
        cyclical_pattern = kwargs.get('cyclical_pattern', None)
        
        cyclical = generate_cyclical_pattern(time_index, cyclical_pattern)
        time_series += cyclical
    
    # Add noise component
    if 'noise' in components:
        noise_type = kwargs.get('noise_type', 'gaussian')
        noise_level = kwargs.get('noise_level', config.get('time_series.default_noise_level', 0.5))
        ar_coefficient = kwargs.get('ar_coefficient', 0.7)
        
        noise = generate_noise(time_index, noise_type, noise_level, ar_coefficient=ar_coefficient)
        time_series += noise
    
    # Add anomalies if requested
    if kwargs.get('add_anomalies', False):
        anomaly_ratio = kwargs.get('anomaly_ratio', 0.05)
        anomaly_type = kwargs.get('anomaly_type', 'point')
        min_anomaly_value = kwargs.get('min_anomaly_value', None)
        max_anomaly_value = kwargs.get('max_anomaly_value', None)
        
        time_series, anomaly_indices = generate_anomalies(
            time_index, time_series, anomaly_ratio, anomaly_type, min_anomaly_value, max_anomaly_value
        )
        
        # Store anomaly indices as an attribute of the time series
        time_series.anomaly_indices = anomaly_indices
    
    # Add missing values if requested
    if kwargs.get('add_missing_values', False):
        missing_ratio = kwargs.get('missing_ratio', 0.05)
        missing_pattern = kwargs.get('missing_pattern', 'random')
        
        time_series = generate_missing_values(time_series, missing_ratio, missing_pattern)
    
    return time_index, time_series

def generate_multivariate_time_series(length: int = None, num_series: int = 3,
                                    correlation_matrix: np.ndarray = None,
                                    start_date: str = '2020-01-01', freq: str = 'D',
                                    include_weekends: bool = True, **kwargs) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Generate a multivariate time series with specified correlations.
    
    Args:
        length (int, optional): Length of the time series
        num_series (int): Number of time series to generate
        correlation_matrix (numpy.ndarray, optional): Correlation matrix for the time series
        start_date (str): Start date in 'YYYY-MM-DD' format
        freq (str): Frequency of the time series ('D' for daily, 'H' for hourly, etc.)
        include_weekends (bool): Whether to include weekends
        **kwargs: Additional arguments for specific components
        
    Returns:
        tuple: (time_index, multivariate_time_series)
            - time_index (pandas.DatetimeIndex): Time index
            - multivariate_time_series (numpy.ndarray): Generated multivariate time series data
    """
    # Use default values from config if not specified
    length = length or config.get('time_series.default_length', 100)
    
    # Generate time index
    time_index = generate_time_index(start_date, length, freq, include_weekends)
    
    # Adjust length if weekends are excluded
    actual_length = len(time_index)
    
    # Generate correlation matrix if not provided
    if correlation_matrix is None:
        # Default to identity matrix (no correlation)
        correlation_matrix = np.eye(num_series)
    
    # Generate independent time series
    independent_series = np.zeros((actual_length, num_series))
    
    for i in range(num_series):
        # Generate independent time series with different parameters
        _, series = generate_time_series(
            length=actual_length,
            components=kwargs.get('components', ['trend', 'seasonality', 'noise']),
            start_date=start_date,
            freq=freq,
            include_weekends=include_weekends,
            trend_coefficient=kwargs.get('trend_coefficient', 0.1) * (i + 1) / num_series,
            seasonality_period=kwargs.get('seasonality_period', 12) * (i % 3 + 1),
            seasonality_amplitude=kwargs.get('seasonality_amplitude', 1.0) * (i % 2 + 1),
            noise_level=kwargs.get('noise_level', 0.5),
            **{k: v for k, v in kwargs.items() if k not in ['trend_coefficient', 'seasonality_period', 'seasonality_amplitude', 'noise_level']}
        )
        
        independent_series[:, i] = series
    
    # Apply Cholesky decomposition to introduce correlations
    L = np.linalg.cholesky(correlation_matrix)
    multivariate_series = independent_series @ L.T
    
    # Add anomalies if requested
    if kwargs.get('add_anomalies', False):
        anomaly_ratio = kwargs.get('anomaly_ratio', 0.05)
        anomaly_type = kwargs.get('anomaly_type', 'point')
        
        # Add anomalies to each series
        anomaly_indices_list = []
        
        for i in range(num_series):
            series_with_anomalies, anomaly_indices = generate_anomalies(
                time_index, multivariate_series[:, i], anomaly_ratio, anomaly_type
            )
            multivariate_series[:, i] = series_with_anomalies
            anomaly_indices_list.append(anomaly_indices)
        
        # Store anomaly indices as an attribute of the multivariate time series
        multivariate_series.anomaly_indices = anomaly_indices_list
    
    # Add missing values if requested
    if kwargs.get('add_missing_values', False):
        missing_ratio = kwargs.get('missing_ratio', 0.05)
        missing_pattern = kwargs.get('missing_pattern', 'random')
        
        # Add missing values to each series
        for i in range(num_series):
            multivariate_series[:, i] = generate_missing_values(
                multivariate_series[:, i], missing_ratio, missing_pattern
            )
    
    return time_index, multivariate_series

def generate_time_series_dataset(num_series: int = 10, length: int = None,
                               output_format: str = 'dataframe', output_file: str = None,
                               multivariate: bool = False, num_variables: int = 3,
                               **kwargs) -> Union[pd.DataFrame, Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]]]:
    """
    Generate a dataset of synthetic time series.
    
    Args:
        num_series (int): Number of time series to generate
        length (int, optional): Length of each time series
        output_format (str): Format of the output ('dataframe', 'dict', 'numpy')
        output_file (str, optional): Path to save the dataset
        multivariate (bool): Whether to generate multivariate time series
        num_variables (int): Number of variables in each multivariate time series
        **kwargs: Additional arguments for time series generation
        
    Returns:
        Union[pandas.DataFrame, dict]: Generated dataset
    """
    # Set random seed if specified in config
    random_seed = config.get('random_seed')
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Generate time series
    dataset = {}
    
    for i in range(num_series):
        if multivariate:
            # Generate multivariate time series
            time_index, series = generate_multivariate_time_series(
                length=length,
                num_series=num_variables,
                **kwargs
            )
            dataset[f'series_{i+1}'] = (time_index, series)
        else:
            # Generate univariate time series
            time_index, series = generate_time_series(
                length=length,
                **kwargs
            )
            dataset[f'series_{i+1}'] = (time_index, series)
    
    # Convert to the requested output format
    if output_format == 'dataframe':
        # Convert to DataFrame
        df_dict = {}
        
        for name, (time_index, series) in dataset.items():
            if multivariate:
                # Multivariate series
                for j in range(series.shape[1]):
                    df_dict[f'{name}_var_{j+1}'] = pd.Series(series[:, j], index=time_index)
            else:
                # Univariate series
                df_dict[name] = pd.Series(series, index=time_index)
        
        result = pd.DataFrame(df_dict)
    
    elif output_format == 'numpy':
        # Convert to numpy arrays
        result = {name: series for name, (_, series) in dataset.items()}
    
    else:
        # Return the dictionary with (time_index, series) tuples
        result = dataset
    
    # Save dataset if output file is specified
    if output_file:
        import os
        from ..utils import exporters
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        file_ext = output_file.split('.')[-1].lower()
        
        if output_format == 'dataframe':
            if file_ext == 'csv':
                exporters.to_csv(result, output_file)
            elif file_ext in ['xls', 'xlsx']:
                exporters.to_excel(result, output_file)
            elif file_ext == 'json':
                exporters.to_json(result, output_file)
            elif file_ext == 'pkl':
                exporters.to_pickle(result, output_file)
            else:
                # Default to CSV
                exporters.to_csv(result, output_file)
        else:
            # For other formats, save as pickle
            exporters.to_pickle(result, output_file)
    
    return result
