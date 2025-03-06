"""
Example script for generating synthetic tabular data.
"""

import os
import sys
import pandas as pd

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules directly
from data_types import tabular_data
from utils import exporters

def main():
    # Example 1: Generate a simple dataset with default settings
    print("Generating a simple tabular dataset...")
    df = tabular_data.generate_dataset(rows=100)
    print(f"Generated dataset with shape: {df.shape}")
    print(df.head())
    print("\n" + "-"*80 + "\n")
    
    # Example 2: Generate a dataset with specific schema
    print("Generating a dataset with custom schema...")
    schema = {
        'id': {'type': 'id', 'start': 1000},
        'name': {'type': 'name'},
        'age': {'type': 'int', 'min': 18, 'max': 65},
        'income': {'type': 'float', 'min': 20000, 'max': 100000, 'distribution': 'normal', 'mean': 50000, 'std': 15000},
        'is_customer': {'type': 'boolean', 'true_probability': 0.7},
        'category': {'type': 'category', 'categories': ['A', 'B', 'C', 'D'], 'weights': [0.4, 0.3, 0.2, 0.1]},
        'date_joined': {'type': 'date', 'start': '2020-01-01', 'end': '2023-12-31'},
        'email': {'type': 'email', 'domain': 'example.com'}
    }
    
    df_custom = tabular_data.generate_dataset(rows=100, schema=schema)
    print(f"Generated custom dataset with shape: {df_custom.shape}")
    print(df_custom.head())
    print("\n" + "-"*80 + "\n")
    
    # Example 3: Generate a dataset with correlations
    print("Generating a dataset with correlations...")
    schema_corr = {
        'age': {'type': 'int', 'min': 18, 'max': 80},
        'income': {'type': 'float', 'min': 20000, 'max': 150000},
        'education_years': {'type': 'int', 'min': 10, 'max': 22},
        'debt': {'type': 'float', 'min': 0, 'max': 100000},
        'savings': {'type': 'float', 'min': 0, 'max': 200000}
    }
    
    correlations = {
        'age': {'income': 0.6, 'education_years': 0.4, 'savings': 0.5},
        'income': {'education_years': 0.7, 'debt': 0.3, 'savings': 0.6},
        'education_years': {'debt': -0.2, 'savings': 0.4},
        'debt': {'savings': -0.5}
    }
    
    df_corr = tabular_data.generate_dataset(rows=100, schema=schema_corr, correlations=correlations)
    print(f"Generated correlated dataset with shape: {df_corr.shape}")
    print(df_corr.head())
    
    # Calculate and print the correlation matrix to verify
    corr_matrix = df_corr.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    print("\n" + "-"*80 + "\n")
    
    # Example 4: Export the dataset to different formats
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'tabular')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting datasets to {output_dir}...")
    
    # Export to CSV
    csv_path = os.path.join(output_dir, 'sample_tabular.csv')
    exporters.to_csv(df_custom, csv_path)
    print(f"Exported to CSV: {csv_path}")
    
    # Export to Excel
    excel_path = os.path.join(output_dir, 'sample_tabular.xlsx')
    exporters.to_excel(df_custom, excel_path)
    print(f"Exported to Excel: {excel_path}")
    
    # Export to JSON
    json_path = os.path.join(output_dir, 'sample_tabular.json')
    exporters.to_json(df_custom, json_path)
    print(f"Exported to JSON: {json_path}")
    
    # Export to Pickle
    pickle_path = os.path.join(output_dir, 'sample_tabular.pkl')
    exporters.to_pickle(df_custom, pickle_path)
    print(f"Exported to Pickle: {pickle_path}")
    
    print("\nTabular data generation examples completed successfully!")

if __name__ == "__main__":
    main()
