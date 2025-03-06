"""
Example script for generating synthetic time series data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules directly
from data_types import time_series_data
from utils import exporters

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'time_series')
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Generate a simple time series with different components
    print("Generating a simple time series with different components...")
    
    # Generate time index
    time_index = time_series_data.generate_time_index(
        start_date='2023-01-01',
        periods=365,
        freq='D'
    )
    
    # Generate individual components
    trend = time_series_data.generate_trend(
        time_index,
        trend_type='linear',
        coefficient=0.05,
        start_value=10
    )
    
    seasonality = time_series_data.generate_seasonality(
        time_index,
        period=365/4,  # Quarterly seasonality
        amplitude=5.0
    )
    
    cyclical = time_series_data.generate_cyclical_pattern(
        time_index,
        pattern=[0, 1, 2, 3, 2, 1]
    )
    
    noise = time_series_data.generate_noise(
        time_index,
        noise_type='gaussian',
        noise_level=1.0
    )
    
    # Combine components
    time_series = trend + seasonality + cyclical + noise
    
    # Plot the components
    plt.figure(figsize=(12, 10))
    
    plt.subplot(5, 1, 1)
    plt.plot(time_index, time_series)
    plt.title('Combined Time Series')
    plt.grid(True)
    
    plt.subplot(5, 1, 2)
    plt.plot(time_index, trend)
    plt.title('Trend Component')
    plt.grid(True)
    
    plt.subplot(5, 1, 3)
    plt.plot(time_index, seasonality)
    plt.title('Seasonal Component')
    plt.grid(True)
    
    plt.subplot(5, 1, 4)
    plt.plot(time_index, cyclical)
    plt.title('Cyclical Component')
    plt.grid(True)
    
    plt.subplot(5, 1, 5)
    plt.plot(time_index, noise)
    plt.title('Noise Component')
    plt.grid(True)
    
    plt.tight_layout()
    components_plot_path = os.path.join(output_dir, 'time_series_components.png')
    plt.savefig(components_plot_path)
    plt.close()
    
    print(f"Time series components plot saved to: {components_plot_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 2: Generate a time series with a specific trend and seasonality
    print("Generating a time series with a specific trend and seasonality...")
    
    # Generate a time series with specific components
    time_index, ts = time_series_data.generate_time_series(
        length=500,
        components=['trend', 'seasonality', 'noise'],
        trend_type='exponential',
        trend_coefficient=0.001,
        trend_start_value=100,
        seasonality_period=50,
        seasonality_amplitude=10,
        noise_type='gaussian',
        noise_level=5
    )
    
    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(time_index, ts)
    plt.title('Time Series with Exponential Trend and Seasonality')
    plt.grid(True)
    plt.tight_layout()
    
    ts_plot_path = os.path.join(output_dir, 'time_series_trend_seasonality.png')
    plt.savefig(ts_plot_path)
    plt.close()
    
    print(f"Time series plot saved to: {ts_plot_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 3: Generate a time series with anomalies
    print("Generating a time series with anomalies...")
    
    # Generate a base time series
    time_index, base_ts = time_series_data.generate_time_series(
        length=300,
        components=['trend', 'seasonality', 'noise'],
        trend_type='linear',
        trend_coefficient=0.05,
        seasonality_period=30,
        noise_level=0.5
    )
    
    # Add anomalies
    ts_with_anomalies, anomaly_indices = time_series_data.generate_anomalies(
        time_index,
        base_ts,
        anomaly_ratio=0.05,
        anomaly_type='point'
    )
    
    # Plot the time series with anomalies
    plt.figure(figsize=(12, 6))
    plt.plot(time_index, ts_with_anomalies, 'b-', label='Time Series')
    
    # Highlight anomalies
    plt.scatter(
        [time_index[i] for i in anomaly_indices],
        [ts_with_anomalies[i] for i in anomaly_indices],
        color='red',
        marker='o',
        label='Anomalies'
    )
    
    plt.title('Time Series with Anomalies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    anomalies_plot_path = os.path.join(output_dir, 'time_series_anomalies.png')
    plt.savefig(anomalies_plot_path)
    plt.close()
    
    print(f"Time series with anomalies plot saved to: {anomalies_plot_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 4: Generate a time series with missing values
    print("Generating a time series with missing values...")
    
    # Generate a base time series
    time_index, base_ts = time_series_data.generate_time_series(
        length=300,
        components=['trend', 'seasonality', 'noise'],
        trend_type='linear',
        trend_coefficient=0.05,
        seasonality_period=30,
        noise_level=0.5
    )
    
    # Add missing values
    ts_with_missing = time_series_data.generate_missing_values(
        base_ts,
        missing_ratio=0.1,
        missing_pattern='random'
    )
    
    # Plot the time series with missing values
    plt.figure(figsize=(12, 6))
    
    # Plot the original time series
    plt.plot(time_index, base_ts, 'b-', alpha=0.5, label='Original')
    
    # Plot the time series with missing values
    plt.plot(time_index, ts_with_missing, 'r-', label='With Missing Values')
    
    plt.title('Time Series with Missing Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    missing_plot_path = os.path.join(output_dir, 'time_series_missing.png')
    plt.savefig(missing_plot_path)
    plt.close()
    
    print(f"Time series with missing values plot saved to: {missing_plot_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 5: Generate a multivariate time series
    print("Generating a multivariate time series...")
    
    # Define a correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.7, -0.3],
        [0.7, 1.0, 0.2],
        [-0.3, 0.2, 1.0]
    ])
    
    # Generate multivariate time series
    time_index, multivariate_ts = time_series_data.generate_multivariate_time_series(
        length=200,
        num_series=3,
        correlation_matrix=correlation_matrix,
        start_date='2023-01-01',
        freq='D'
    )
    
    # Plot the multivariate time series
    plt.figure(figsize=(12, 8))
    
    for i in range(multivariate_ts.shape[1]):
        plt.subplot(3, 1, i+1)
        plt.plot(time_index, multivariate_ts[:, i])
        plt.title(f'Variable {i+1}')
        plt.grid(True)
    
    plt.tight_layout()
    multivariate_plot_path = os.path.join(output_dir, 'multivariate_time_series.png')
    plt.savefig(multivariate_plot_path)
    plt.close()
    
    print(f"Multivariate time series plot saved to: {multivariate_plot_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 6: Generate a time series dataset
    print("Generating a time series dataset...")
    
    # Generate a dataset of time series
    dataset = time_series_data.generate_time_series_dataset(
        num_series=5,
        length=100,
        output_format='dataframe',
        multivariate=False
    )
    
    # Save the dataset to a CSV file
    dataset_path = os.path.join(output_dir, 'time_series_dataset.csv')
    exporters.to_csv(dataset, dataset_path)
    
    print(f"Generated a dataset with {len(dataset.columns)} time series")
    print(f"Dataset saved to: {dataset_path}")
    
    print("\nTime series data generation examples completed successfully!")

if __name__ == "__main__":
    main()
