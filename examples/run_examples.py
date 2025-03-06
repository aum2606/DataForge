"""
Script to run the synthetic data generator examples.
This script handles the import paths correctly.
"""

import os
import sys
import importlib

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Create necessary directories
output_dir = os.path.join(parent_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

def run_example(example_name):
    """Run a specific example."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXAMPLE: {example_name}")
    print(f"{'='*80}\n")
    
    # Import the example module dynamically
    try:
        # First try importing as a module
        module_name = f"examples.{example_name}"
        example = importlib.import_module(module_name)
        
        # Run the main function
        if hasattr(example, 'main'):
            example.main()
        else:
            print(f"Error: No main() function found in {module_name}")
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")

def main():
    # List of examples to run
    examples = [
        "generate_tabular",
        "generate_image",
        "generate_text",
        "generate_timeseries",
        "generate_audio",
        "generate_all_types"
    ]
    
    # Ask the user which example to run
    print("Available examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    choice = input("\nEnter the number of the example to run (or 'all' to run all): ")
    
    if choice.lower() == 'all':
        # Run all examples
        for example in examples:
            run_example(example)
    else:
        try:
            index = int(choice) - 1
            if 0 <= index < len(examples):
                run_example(examples[index])
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(examples)}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'all'.")

if __name__ == "__main__":
    main()
