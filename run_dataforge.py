#!/usr/bin/env python
"""
DataForge Web Interface Runner

This script ensures Flask is installed and runs the DataForge web interface.
"""

import subprocess
import sys
import os

def check_flask_installed():
    """Check if Flask is installed, and install it if not."""
    try:
        import flask
        print("Flask is already installed.")
        return True
    except ImportError:
        print("Flask is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
            print("Flask has been successfully installed.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install Flask. Please install it manually using 'pip install flask'.")
            return False

def run_flask_app():
    """Run the Flask application."""
    try:
        print("Starting DataForge web interface...")
        subprocess.run([sys.executable, "app.py"], check=True)
    except subprocess.CalledProcessError:
        print("Failed to start the Flask application. Make sure 'app.py' exists in the current directory.")
        return False
    return True

def main():
    """Main function to set up and run the DataForge web interface."""
    print("=" * 80)
    print("DataForge Web Interface Setup")
    print("=" * 80)
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("Warning: It is recommended to run this in a virtual environment.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please activate your virtual environment and try again.")
            sys.exit(0)
    
    # Check if Flask is installed
    if not check_flask_installed():
        sys.exit(1)
    
    # Run the Flask application
    if run_flask_app():
        print("\nDataForge web interface has been started.")
        print("Open your browser and navigate to: http://localhost:5000")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 