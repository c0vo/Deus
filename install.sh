#!/bin/bash

echo "======================================"
echo " Deus - Termux Setup"
echo "======================================"

# Update and upgrade packages
echo "[1/5] Updating package list..."
pkg update -y && pkg upgrade -y

# Install Python and essential build tools.
# termux-api ships termux-wake-lock — without it Android's Doze suspends the
# process when the screen turns off and the dashboard stops answering.
echo "[2/5] Installing Python and build dependencies..."
pkg install -y python python-pip rust binutils cmake clang termux-api

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
echo "   termux-wake-lock"
echo "   python main.py"
echo ""
echo " IMPORTANT — for 24/7 operation you must also:"
echo "   1. Install the Termux:API app from F-Droid (the termux-api"
echo "      package alone is only the CLI half)."
echo "   2. Exempt Termux from battery optimization in Android settings."
echo "   Without both, Doze freezes the process and the dashboard"
echo "   times out from other devices on the LAN."
echo "======================================"
