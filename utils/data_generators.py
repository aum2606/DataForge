"""
Common functions to generate random data for various data types.
"""

import random
import string
import numpy as np
from faker import Faker

fake = Faker()

def generate_random_number(min_val=0, max_val=100, data_type='float'):
    """
    Generate a random number within the specified range.
    
    Args:
        min_val (float): Minimum value
        max_val (float): Maximum value
        data_type (str): Type of data to generate ('int' or 'float')
        
    Returns:
        int or float: Random number
    """
    if data_type == 'int':
        return random.randint(min_val, max_val)
    else:
        return random.uniform(min_val, max_val)

def generate_random_string(length=10, include_digits=True, include_special=False):
    """
    Generate a random string of specified length.
    
    Args:
        length (int): Length of the string
        include_digits (bool): Whether to include digits
        include_special (bool): Whether to include special characters
        
    Returns:
        str: Random string
    """
    chars = string.ascii_letters
    if include_digits:
        chars += string.digits
    if include_special:
        chars += string.punctuation
    
    return ''.join(random.choice(chars) for _ in range(length))

def generate_random_date(start_date='1970-01-01', end_date='2023-12-31', date_format='%Y-%m-%d'):
    """
    Generate a random date within the specified range.
    
    Args:
        start_date (str): Start date in the specified format
        end_date (str): End date in the specified format
        date_format (str): Date format
        
    Returns:
        str: Random date in the specified format
    """
    return fake.date_between_dates(start_date=start_date, end_date=end_date).strftime(date_format)

def generate_random_name():
    """
    Generate a random full name.
    
    Returns:
        str: Random name
    """
    return fake.name()

def generate_random_email(name=None):
    """
    Generate a random email address.
    
    Args:
        name (str, optional): Name to use for the email address
        
    Returns:
        str: Random email address
    """
    if name:
        return fake.email(name=name)
    return fake.email()

def generate_random_address():
    """
    Generate a random address.
    
    Returns:
        str: Random address
    """
    return fake.address()

def generate_random_phone_number():
    """
    Generate a random phone number.
    
    Returns:
        str: Random phone number
    """
    return fake.phone_number()

def generate_random_text(min_words=10, max_words=100):
    """
    Generate random text with the specified number of words.
    
    Args:
        min_words (int): Minimum number of words
        max_words (int): Maximum number of words
        
    Returns:
        str: Random text
    """
    num_words = random.randint(min_words, max_words)
    return fake.text(max_nb_chars=num_words * 7)  # Approximate 7 chars per word on average

def generate_random_color(as_rgb=True, as_hex=False):
    """
    Generate a random color.
    
    Args:
        as_rgb (bool): Whether to return the color as RGB tuple
        as_hex (bool): Whether to return the color as hex string
        
    Returns:
        tuple or str: Random color
    """
    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    
    if as_hex:
        return f'#{r:02x}{g:02x}{b:02x}'
    return (r, g, b)

def generate_random_boolean(true_probability=0.5):
    """
    Generate a random boolean value.
    
    Args:
        true_probability (float): Probability of returning True
        
    Returns:
        bool: Random boolean
    """
    return random.random() < true_probability

def generate_random_category(categories, weights=None):
    """
    Generate a random category from the specified list.
    
    Args:
        categories (list): List of categories
        weights (list, optional): List of weights for each category
        
    Returns:
        Any: Random category
    """
    return random.choices(categories, weights=weights, k=1)[0]

def generate_correlated_values(base_value, correlation_factor, min_val=0, max_val=100, noise_level=0.1):
    """
    Generate a value correlated with the base value.
    
    Args:
        base_value (float): Base value to correlate with
        correlation_factor (float): Correlation factor (-1 to 1)
        min_val (float): Minimum value
        max_val (float): Maximum value
        noise_level (float): Level of noise to add
        
    Returns:
        float: Correlated value
    """
    # Normalize base value to 0-1 range
    normalized_base = (base_value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    
    # Apply correlation factor
    if correlation_factor > 0:
        target = normalized_base
    else:
        target = 1 - normalized_base
    
    # Add noise
    noise = random.uniform(-noise_level, noise_level)
    result = target + noise * abs(correlation_factor)
    
    # Clamp to 0-1 range
    result = max(0, min(1, result))
    
    # Scale back to original range
    return min_val + result * (max_val - min_val)
