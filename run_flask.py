#!/usr/bin/env python3
"""
Run script for the DataForge Web Interface

This script starts the Flask application serving the DataForge web interface.
"""

import os
import argparse
from app import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the DataForge web interface')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting DataForge web interface on http://{args.host}:{args.port}")
    print("Press Ctrl+C to quit")
    
    # Create necessary directories
    os.makedirs(os.path.join('static', 'generated_data'), exist_ok=True)
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug) 