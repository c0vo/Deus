#!/bin/bash

echo "======================================"
echo " Project Scrooge V2 - Termux Setup"
echo "======================================"

# Update and upgrade packages
echo "[1/5] Updating package list..."
pkg update -y && pkg upgrade -y

# Install Python and essential build tools
echo "[2/5] Installing Python and build dependencies..."
pkg install -y python python-pip rust binutils cmake clang

# Install pre-built numpy from Termux repos (can't compile from source)
echo "[3/5] Installing pre-built numpy from Termux repos..."
pkg install -y python-numpy

# Set up virtual environment (--system-site-packages to access Termux numpy)
echo "[4/5] Creating Python virtual environment..."
python -m venv --system-site-packages venv
source venv/bin/activate

# Install requirements
echo "[5/5] Installing Python dependencies..."
pip install --upgrade pip

# Set Android API level for Rust-based packages (pydantic-core, etc.)
export ANDROID_API_LEVEL=24

# Install requirements
pip install -r requirements.txt

echo "======================================"
echo " Setup Complete! 🎉"
echo ""
echo " To run the bot, execute the following:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo "======================================"
