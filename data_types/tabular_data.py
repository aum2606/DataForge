"""
Module for generating synthetic tabular data.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from typing import Dict, List, Union, Any, Optional
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules
from utils import data_generators
from utils import distributions
from config import config

fake = Faker()

def generate_column(size: int, column_spec: Dict[str, Any]) -> List:
    """
    Generate a column of data based on the specification.
    
    Args:
        size (int): Number of elements to generate
        column_spec (dict): Specification for the column
            - type (str): Type of data to generate
                - 'int': Integer values
                - 'float': Float values
                - 'boolean': Boolean values
                - 'category': Categorical values
                - 'date': Date values
                - 'datetime': Datetime values
                - 'name': Person names
                - 'address': Addresses
                - 'email': Email addresses
                - 'phone': Phone numbers
                - 'text': Random text
                - 'id': Sequential IDs
            - min (int/float): Minimum value (for numeric types)
            - max (int/float): Maximum value (for numeric types)
            - distribution (str): Distribution to use (for numeric types)
                - 'uniform': Uniform distribution
                - 'normal': Normal distribution
                - 'exponential': Exponential distribution
                - 'lognormal': Log-normal distribution
                - 'poisson': Poisson distribution
            - mean (float): Mean value (for normal distribution)
            - std (float): Standard deviation (for normal distribution)
            - categories (list): List of categories (for categorical data)
            - weights (list): List of weights for categories (for categorical data)
            - true_probability (float): Probability of True (for boolean data)
            - start (str/int): Start date/time or start ID (for date/time or ID data)
            - end (str): End date/time (for date/time data)
            - format (str): Format string (for date/time data)
            - missing_ratio (float): Ratio of missing values (None)
    
    Returns:
        list: Generated column data
    """
    data_type = column_spec.get('type', 'float')
    size = int(size)  # Ensure size is an integer
    
    # Get missing value ratio if specified
    missing_ratio = column_spec.get('missing_ratio', 0.0)
    
    # Generate data based on type
    if data_type == 'int':
        min_val = column_spec.get('min', 0)
        max_val = column_spec.get('max', 100)
        distribution = column_spec.get('distribution', 'uniform')
        
        if distribution == 'uniform':
            data = np.random.randint(min_val, max_val + 1, size=size).tolist()
        elif distribution == 'normal':
            mean = column_spec.get('mean', (min_val + max_val) / 2)
            std = column_spec.get('std', (max_val - min_val) / 6)
            data = np.random.normal(mean, std, size=size).astype(int).tolist()
            # Clip to min/max range
            data = [max(min_val, min(max_val, x)) for x in data]
        else:
            # Default to uniform if distribution not supported
            data = np.random.randint(min_val, max_val + 1, size=size).tolist()
    
    elif data_type == 'float':
        min_val = column_spec.get('min', 0.0)
        max_val = column_spec.get('max', 1.0)
        distribution = column_spec.get('distribution', 'uniform')
        
        if distribution == 'uniform':
            data = np.random.uniform(min_val, max_val, size=size).tolist()
        elif distribution == 'normal':
            mean = column_spec.get('mean', (min_val + max_val) / 2)
            std = column_spec.get('std', (max_val - min_val) / 6)
            data = np.random.normal(mean, std, size=size).tolist()
            # Clip to min/max range
            data = [max(min_val, min(max_val, x)) for x in data]
        elif distribution == 'exponential':
            scale = column_spec.get('scale', 1.0)
            data = np.random.exponential(scale, size=size).tolist()
            # Clip to min/max range
            data = [max(min_val, min(max_val, x)) for x in data]
        elif distribution == 'lognormal':
            mean = column_spec.get('mean', 0)
            sigma = column_spec.get('sigma', 1)
            data = np.random.lognormal(mean, sigma, size=size).tolist()
            # Clip to min/max range
            data = [max(min_val, min(max_val, x)) for x in data]
        else:
            # Default to uniform if distribution not supported
            data = np.random.uniform(min_val, max_val, size=size).tolist()
    
    elif data_type == 'boolean':
        true_prob = column_spec.get('true_probability', 0.5)
        data = np.random.choice([True, False], size=size, p=[true_prob, 1-true_prob]).tolist()
    
    elif data_type == 'category':
        categories = column_spec.get('categories', ['A', 'B', 'C'])
        weights = column_spec.get('weights', None)
        data = np.random.choice(categories, size=size, p=weights).tolist()
    
    elif data_type == 'date':
        start = column_spec.get('start', '2020-01-01')
        end = column_spec.get('end', '2023-12-31')
        date_format = column_spec.get('format', '%Y-%m-%d')
        
        # Convert to datetime objects if strings
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)
        
        # Generate random dates
        data = [pd.to_datetime(np.random.randint(
            start.value // 10**9, 
            end.value // 10**9
        ), unit='s') for _ in range(size)]
        
        # Format dates if format is specified
        if date_format:
            data = [d.strftime(date_format) for d in data]
    
    elif data_type == 'datetime':
        start = column_spec.get('start', '2020-01-01 00:00:00')
        end = column_spec.get('end', '2023-12-31 23:59:59')
        datetime_format = column_spec.get('format', '%Y-%m-%d %H:%M:%S')
        
        # Convert to datetime objects if strings
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)
        
        # Generate random datetimes
        data = [pd.to_datetime(np.random.randint(
            start.value // 10**9, 
            end.value // 10**9
        ), unit='s') for _ in range(size)]
        
        # Format datetimes if format is specified
        if datetime_format:
            data = [d.strftime(datetime_format) for d in data]
    
    elif data_type == 'name':
        data = [fake.name() for _ in range(size)]
    
    elif data_type == 'address':
        data = [fake.address().replace('\n', ', ') for _ in range(size)]
    
    elif data_type == 'email':
        domain = column_spec.get('domain', None)
        if domain:
            data = [f"{fake.user_name()}@{domain}" for _ in range(size)]
        else:
            data = [fake.email() for _ in range(size)]
    
    elif data_type == 'phone':
        data = [fake.phone_number() for _ in range(size)]
    
    elif data_type == 'text':
        min_length = column_spec.get('min_length', 10)
        max_length = column_spec.get('max_length', 100)
        data = [fake.text(random.randint(min_length, max_length)) for _ in range(size)]
    
    elif data_type == 'id':
        start = column_spec.get('start', 1)
        data = list(range(start, start + size))
    
    else:
        # Default to random floats if type not supported
        data = np.random.random(size=size).tolist()
    
    # Apply missing values if specified
    if missing_ratio > 0:
        for i in range(size):
            if random.random() < missing_ratio:
                data[i] = None
    
    return data

def generate_correlated_column(base_column: List, correlation: float, column_spec: Dict[str, Any]) -> List:
    """
    Generate a column that is correlated with a base column.
    
    Args:
        base_column (list): Base column to correlate with
        correlation (float): Correlation coefficient (-1 to 1)
        column_spec (dict): Specification for the column
    
    Returns:
        list: Generated correlated column
    """
    size = len(base_column)
    
    # Generate independent column
    independent_column = generate_column(size, column_spec)
    
    # Handle missing values in base column
    base_column_clean = [0 if x is None else x for x in base_column]
    
    # Standardize base column
    base_mean = np.mean(base_column_clean)
    base_std = np.std(base_column_clean) if np.std(base_column_clean) > 0 else 1
    base_column_std = [(x - base_mean) / base_std for x in base_column_clean]
    
    # Handle missing values and non-numeric data in independent column
    if all(isinstance(x, (int, float)) or x is None for x in independent_column):
        # For numeric columns
        independent_column_clean = [0 if x is None else x for x in independent_column]
        
        # Standardize independent column
        ind_mean = np.mean(independent_column_clean)
        ind_std = np.std(independent_column_clean) if np.std(independent_column_clean) > 0 else 1
        independent_column_std = [(x - ind_mean) / ind_std for x in independent_column_clean]
        
        # Generate correlated column
        correlated_column_std = [correlation * b + np.sqrt(1 - correlation**2) * i 
                                for b, i in zip(base_column_std, independent_column_std)]
        
        # Unstandardize back to original scale
        correlated_column = [i * ind_std + ind_mean for i in correlated_column_std]
        
        # Apply original data type
        if all(isinstance(x, int) or x is None for x in independent_column):
            correlated_column = [int(round(x)) for x in correlated_column]
        
        # Restore missing values
        for i in range(size):
            if independent_column[i] is None:
                correlated_column[i] = None
        
        return correlated_column
    else:
        # For non-numeric columns, just return the independent column
        # (correlation not applicable)
        return independent_column

def generate_dataset(rows: int = 1000, schema: Dict[str, Dict[str, Any]] = None, 
                     correlations: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
    """
    Generate a synthetic tabular dataset.
    
    Args:
        rows (int): Number of rows to generate
        schema (dict): Schema definition for the dataset
            - Each key is a column name
            - Each value is a column specification dict (see generate_column)
        correlations (dict): Correlation specifications
            - Each key is a column name
            - Each value is a dict mapping other column names to correlation coefficients
    
    Returns:
        pandas.DataFrame: Generated dataset
    """
    if schema is None:
        # Default schema if none provided
        schema = {
            'id': {'type': 'id', 'start': 1},
            'value': {'type': 'float', 'min': 0, 'max': 100, 'distribution': 'normal'},
            'category': {'type': 'category', 'categories': ['A', 'B', 'C']},
            'date': {'type': 'date'},
            'name': {'type': 'name'}
        }
    
    if correlations is None:
        correlations = {}
    
    # Set random seed if specified in config
    random_seed = config.get('random_seed')
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        fake.seed_instance(random_seed)
    
    # Generate independent columns first
    data = {}
    for column_name, column_spec in schema.items():
        # Skip columns that are correlated with others
        if column_name in correlations:
            continue
        
        data[column_name] = generate_column(rows, column_spec)
    
    # Generate correlated columns
    for column_name, corr_dict in correlations.items():
        if column_name not in schema:
            continue
        
        # Find the base column with the highest absolute correlation
        base_column_name = None
        max_abs_corr = 0
        
        for other_column, corr in corr_dict.items():
            if other_column in data and abs(corr) > max_abs_corr:
                base_column_name = other_column
                max_abs_corr = abs(corr)
        
        if base_column_name is not None:
            # Generate correlated column
            data[column_name] = generate_correlated_column(
                data[base_column_name],
                corr_dict[base_column_name],
                schema[column_name]
            )
        else:
            # If no base column found, generate independent column
            data[column_name] = generate_column(rows, schema[column_name])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure default schema is properly handled
    if schema == {
        'id': {'type': 'id', 'start': 1},
        'value': {'type': 'float', 'min': 0, 'max': 100, 'distribution': 'normal'},
        'category': {'type': 'category', 'categories': ['A', 'B', 'C']},
        'date': {'type': 'date'},
        'name': {'type': 'name'}
    }:
        df['id'] = df['id'].astype(int)
        df['value'] = df['value'].astype(float)
        df['category'] = df['category'].astype('category')
        df['date'] = pd.to_datetime(df['date'])
        df['name'] = df['name'].astype(str)
    
    return df

def generate_from_template(template_path: str, rows: int = None, output_path: str = None) -> pd.DataFrame:
    """
    Generate a dataset based on a template file.
    
    Args:
        template_path (str): Path to the template file (JSON)
        rows (int, optional): Number of rows to generate (overrides template)
        output_path (str, optional): Path to save the generated dataset
    
    Returns:
        pandas.DataFrame: Generated dataset
    """
    import json
    
    # Load template
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    # Extract parameters from template
    template_rows = template.get('rows', 1000)
    rows = rows or template_rows
    
    schema = template.get('schema', {})
    correlations = template.get('correlations', {})
    
    # Generate dataset
    df = generate_dataset(rows, schema, correlations)
    
    # Save if output path is provided
    if output_path:
        from utils import exporters
        
        file_ext = output_path.split('.')[-1].lower()
        if file_ext == 'csv':
            exporters.to_csv(df, output_path)
        elif file_ext in ['xls', 'xlsx']:
            exporters.to_excel(df, output_path)
        elif file_ext == 'json':
            exporters.to_json(df, output_path)
        elif file_ext == 'pkl':
            exporters.to_pickle(df, output_path)
        else:
            # Default to CSV
            exporters.to_csv(df, output_path)
    
    return df
