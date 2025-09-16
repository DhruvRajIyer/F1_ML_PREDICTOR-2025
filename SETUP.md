# F1 Predictions App - Setup Guide

This guide will help you set up and run the F1 Predictions Streamlit application with a clean, simplified environment.

## Quick Setup

For a quick setup, simply run the provided setup script:

```bash
./setup.sh
```

This script will:
1. Create a new virtual environment called `env`
2. Install all required dependencies
3. Provide instructions for running the app

## Manual Setup

If you prefer to set up manually, follow these steps:

### 1. Create a Virtual Environment

```bash
# Create a new virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

## Running the App

After setting up the environment, you can run the app using:

```bash
# Make sure the virtual environment is activated
source env/bin/activate

# Run the app
python run_app.py
```

By default, the app will run on port 8502. You can specify a different port using:

```bash
python run_app.py --port 8505
```

## Troubleshooting

### Port Already in Use

If you see an error that the port is already in use, try running the app with a different port:

```bash
python run_app.py --port 8503
```

### Missing Dependencies

If you encounter any missing dependency errors, make sure your virtual environment is activated and try reinstalling the requirements:

```bash
source env/bin/activate
pip install -r requirements.txt
```

### FastF1 Cache Issues

If you encounter issues with FastF1 data loading, try clearing the cache:

```bash
rm -rf f1_cache
```

## Project Structure

The project has been modularized for better organization:

```
/app
├── data/         # Data loading and processing
├── models/       # Prediction models
├── ui/           # UI components
└── visualization/ # Visualization functions
```

This structure makes the code more maintainable and easier to extend.
