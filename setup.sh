#!/bin/bash

# UAV Localization System Setup Script

echo "=== UAV Localization System Setup ==="

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To run the demo:"
echo "  source .venv/bin/activate"
echo "  python demo.py"
echo ""
echo "To run tests:"
echo "  source .venv/bin/activate"
echo "  python tests/test_uav_localization.py"
