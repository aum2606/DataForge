"""
Script to run example files from the synthetic data generator project.
This script sets up the correct import paths and then runs the specified example.
"""

import os
import sys
import importlib.util
import argparse

def run_example(example_name):
    """Run a specific example script."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to the Python path
    sys.path.insert(0, project_root)
    
    # Construct the path to the example script
    example_path = os.path.join(project_root, 'examples', f'generate_{example_name}.py')
    
    if not os.path.exists(example_path):
        print(f"Error: Example '{example_name}' not found at {example_path}")
        print("Available examples:")
        for file in os.listdir(os.path.join(project_root, 'examples')):
            if file.startswith('generate_') and file.endswith('.py'):
                print(f"  - {file[9:-3]}")
        return
    
    # Load the example module
    spec = importlib.util.spec_from_file_location(f"example_{example_name}", example_path)
    example_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_module)
    
    # Run the main function if it exists
    if hasattr(example_module, 'main'):
        example_module.main()
    else:
        print(f"Warning: No main() function found in {example_path}")

def main():
    parser = argparse.ArgumentParser(description='Run synthetic data generator examples')
    parser.add_argument('example', nargs='?', default='all_types',
                        help='Name of the example to run (without the "generate_" prefix and ".py" suffix)')
    args = parser.parse_args()
    
    run_example(args.example)

if __name__ == "__main__":
    main()
