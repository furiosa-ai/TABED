#!/bin/bash
# TABED Installation Script
# This script creates a virtual environment using uv for fast, reproducible setup.
#
# Usage:
#   ./install.sh
#
# After installation, activate the environment with:
#   source .venv/bin/activate

set -e

echo "==================================="
echo "TABED Environment Installation"
echo "==================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed successfully."
fi

echo ""
echo "Creating virtual environment..."
uv venv .venv --python 3.10

echo "Activating the virtual environment..."
source .venv/bin/activate

echo ""
echo "Installing dependencies from requirements.txt..."
echo "This may take a few minutes..."
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install flash-attn==2.6.2 --no-build-isolation
uv pip install -r requirements.txt

echo ""
echo "==================================="
echo "Installation complete!"
echo "==================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To set your Hugging Face token:"
echo "  export HF_TOKEN=\"your_huggingface_token\""