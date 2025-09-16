#!/bin/bash
# F1 Predictions - Environment Setup Script

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== F1 Predictions - Environment Setup ===${NC}"
echo "This script will set up your Python environment for the F1 Predictions project."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo -e "\n${BLUE}Creating virtual environment...${NC}"
    python3 -m venv env
    echo -e "${GREEN}Virtual environment created successfully!${NC}"
else
    echo -e "\n${BLUE}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "\n${BLUE}Activating virtual environment...${NC}"
source env/bin/activate

# Install dependencies
echo -e "\n${BLUE}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "\n${GREEN}Setup complete! Your environment is ready.${NC}"
echo -e "To activate the environment in the future, run: ${BLUE}source env/bin/activate${NC}"
echo -e "To run the F1 Predictions app, run: ${BLUE}python run_app.py${NC}"
echo -e "To deactivate the environment when done, run: ${BLUE}deactivate${NC}"
