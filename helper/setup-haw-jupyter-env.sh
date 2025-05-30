#!/bin/bash

# Update apt and install swig
echo "Updating apt and installing swig..."
sudo apt update && sudo apt install swig -y

# Uninstall existing PyTorch installations
echo "Uninstalling existing PyTorch, torchvision, and torchaudio..."
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with cu127 support
echo "Installing PyTorch, torchvision, and torchaudio with cu127 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu127

# Install remaining requirements from requirements.txt
echo "Installing remaining requirements from requirements.txt..."
pip install -r requirements.txt

echo "Script execution complete."