#!/usr/bin/env python3
"""
F1 Predictions - Streamlit App Runner

This script runs the Streamlit app with proper environment setup.
"""

import os
import sys
import subprocess
import argparse

def main():
    """
    Main function to run the Streamlit app
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run F1 Predictions Streamlit App")
    parser.add_argument(
        "-p", "--port", 
        type=int, 
        default=8502, 
        help="Port to run Streamlit on (default: 8502)"
    )
    args = parser.parse_args()
    
    print(f"🏎️ Starting F1 AI Race Predictor on port {args.port}...")
    print("=" * 50)
    
    # Run the Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(args.port),
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nApp stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
