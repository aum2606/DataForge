"""
Configuration settings for the synthetic data generator.
"""

import os
import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    # General settings
    'random_seed': 42,  # Set to None for truly random data
    'output_dir': 'output',
    
    # Tabular data settings
    'tabular': {
        'default_rows': 1000,
        'max_rows': 1000000,
        'default_missing_value_ratio': 0.05,
        'correlation_matrix_sparsity': 0.7,
    },
    
    # Image data settings
    'image': {
        'default_width': 256,
        'default_height': 256,
        'default_channels': 3,
        'default_format': 'PNG',
        'max_dimension': 4096,
    },
    
    # Text data settings
    'text': {
        'default_min_length': 10,
        'default_max_length': 1000,
        'default_language': 'en',
        'supported_languages': ['en', 'es', 'fr', 'de', 'it'],
    },
    
    # Time series data settings
    'time_series': {
        'default_length': 100,
        'default_frequency': 'D',  # Daily
        'supported_frequencies': ['D', 'W', 'M', 'Q', 'Y', 'H', 'T', 'S'],
        'default_trend_coefficient': 0.1,
        'default_seasonality_period': 12,
        'default_noise_level': 0.5,
    },
    
    # Audio data settings
    'audio': {
        'default_duration': 5.0,  # seconds
        'default_sample_rate': 44100,  # Hz
        'max_duration': 60.0,  # seconds
        'default_format': 'WAV',
    }
}

class Config:
    """Configuration manager for the synthetic data generator."""
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str, optional): Path to a custom configuration file
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            
            # Update the default configuration with custom values
            self._update_dict(self.config, custom_config)
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
    
    def save_config(self, config_path):
        """
        Save the current configuration to a JSON file.
        
        Args:
            config_path (str): Path to save the configuration file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving configuration to {config_path}: {e}")
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): Configuration key (can be nested using dots, e.g., 'tabular.default_rows')
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or the default value if not found
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key (str): Configuration key (can be nested using dots, e.g., 'tabular.default_rows')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def _update_dict(self, target, source):
        """
        Recursively update a dictionary.
        
        Args:
            target (dict): Dictionary to update
            source (dict): Dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value

# Create a global configuration instance
config = Config()

def get_config():
    """
    Get the global configuration instance.
    
    Returns:
        Config: Global configuration instance
    """
    return config

def load_config(config_path):
    """
    Load configuration from a file into the global configuration instance.
    
    Args:
        config_path (str): Path to the configuration file
    """
    config.load_config(config_path)

def save_config(config_path):
    """
    Save the global configuration to a file.
    
    Args:
        config_path (str): Path to save the configuration file
    """
    config.save_config(config_path)

def get(key, default=None):
    """
    Get a configuration value from the global configuration.
    
    Args:
        key (str): Configuration key
        default: Default value to return if the key is not found
        
    Returns:
        The configuration value or the default value if not found
    """
    return config.get(key, default)

def set(key, value):
    """
    Set a configuration value in the global configuration.
    
    Args:
        key (str): Configuration key
        value: Value to set
    """
    config.set(key, value)
