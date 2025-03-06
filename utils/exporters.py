"""
Utility functions for exporting generated data in different formats.
"""

import os
import json
import csv
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import soundfile as sf

def to_csv(data, filepath, index=False, **kwargs):
    """
    Export data to a CSV file.
    
    Args:
        data (pandas.DataFrame or dict): Data to export
        filepath (str): Path to the output file
        index (bool): Whether to include index in the output
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_csv()
    """
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Export to CSV
    data.to_csv(filepath, index=index, **kwargs)
    
    return filepath

def to_json(data, filepath, orient='records', **kwargs):
    """
    Export data to a JSON file.
    
    Args:
        data (pandas.DataFrame, dict, or list): Data to export
        filepath (str): Path to the output file
        orient (str): Orientation of the JSON file if data is a DataFrame
        **kwargs: Additional arguments to pass to json.dump()
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Convert DataFrame to dict if necessary
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient=orient)
    
    # Export to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, **kwargs)
    
    return filepath

def to_pickle(data, filepath, **kwargs):
    """
    Export data to a pickle file.
    
    Args:
        data (any): Data to export
        filepath (str): Path to the output file
        **kwargs: Additional arguments to pass to pickle.dump()
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Export to pickle
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, **kwargs)
    
    return filepath

def to_excel(data, filepath, index=False, **kwargs):
    """
    Export data to an Excel file.
    
    Args:
        data (pandas.DataFrame or dict): Data to export
        filepath (str): Path to the output file
        index (bool): Whether to include index in the output
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_excel()
    """
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Export to Excel
    data.to_excel(filepath, index=index, **kwargs)
    
    return filepath

def to_image(data, filepath, format='PNG', **kwargs):
    """
    Export image data to an image file.
    
    Args:
        data (numpy.ndarray): Image data (HxWxC or HxW)
        filepath (str): Path to the output file
        format (str): Image format (PNG, JPEG, etc.)
        **kwargs: Additional arguments to pass to PIL.Image.save()
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Convert to uint8 if necessary
    if data.dtype != np.uint8:
        if data.max() <= 1.0:
            data = (data * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(data)
    
    # Export to image file
    img.save(filepath, format=format, **kwargs)
    
    return filepath

def to_audio(data, filepath, sample_rate=44100, **kwargs):
    """
    Export audio data to an audio file.
    
    Args:
        data (numpy.ndarray): Audio data
        filepath (str): Path to the output file
        sample_rate (int): Sample rate of the audio
        **kwargs: Additional arguments to pass to soundfile.write()
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Export to audio file
    sf.write(filepath, data, sample_rate, **kwargs)
    
    return filepath

def to_plot(data, filepath, plot_type='line', figsize=(10, 6), dpi=100, **kwargs):
    """
    Export data as a plot image.
    
    Args:
        data (pandas.DataFrame, dict, or numpy.ndarray): Data to plot
        filepath (str): Path to the output file
        plot_type (str): Type of plot ('line', 'scatter', 'bar', 'hist', etc.)
        figsize (tuple): Figure size
        dpi (int): DPI of the output image
        **kwargs: Additional arguments to pass to the plotting function
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create plot based on plot_type
    if plot_type == 'line':
        if isinstance(data, pd.DataFrame):
            data.plot(kind='line', **kwargs)
        else:
            plt.plot(data, **kwargs)
    elif plot_type == 'scatter':
        if isinstance(data, pd.DataFrame):
            data.plot(kind='scatter', **kwargs)
        else:
            plt.scatter(range(len(data)), data, **kwargs)
    elif plot_type == 'bar':
        if isinstance(data, pd.DataFrame):
            data.plot(kind='bar', **kwargs)
        else:
            plt.bar(range(len(data)), data, **kwargs)
    elif plot_type == 'hist':
        if isinstance(data, pd.DataFrame):
            data.plot(kind='hist', **kwargs)
        else:
            plt.hist(data, **kwargs)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi)
    plt.close()
    
    return filepath

def to_html(data, filepath, **kwargs):
    """
    Export data to an HTML file.
    
    Args:
        data (pandas.DataFrame or dict): Data to export
        filepath (str): Path to the output file
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_html()
    """
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Export to HTML
    html = data.to_html(**kwargs)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return filepath
