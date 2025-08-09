#!/bin/bash

# Setup script for mini2 to ensure it has the required environment
# Run this once on mini2 to set up the Python environment

set -e

echo "Setting up mini2 for distributed MLX inference..."
echo ""

# Check if we're on mini2
HOSTNAME=$(hostname -s)
if [ "$HOSTNAME" != "mini2" ]; then
    echo "This script should be run on mini2, not $HOSTNAME"
    echo "Copy and run it on mini2:"
    echo "  scp setup_mini2.sh 192.168.5.2:/Users/mini2/"
    echo "  ssh 192.168.5.2 'bash /Users/mini2/setup_mini2.sh'"
    exit 1
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv if it doesn't exist
if [ ! -d "/Users/mini2/.venv" ]; then
    echo "Creating Python 3.13.5 virtual environment..."
    uv venv --python 3.13.5 /Users/mini2/.venv
fi

# Activate venv
source /Users/mini2/.venv/bin/activate

# Install required packages
echo "Installing required packages..."
uv pip install \
    mlx>=0.21.0 \
    mlx-lm>=0.20.0 \
    mpi4py>=3.1.0 \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.34.0 \
    pydantic>=2.10.0 \
    psutil>=6.1.0 \
    numpy>=1.26.0

echo ""
echo "✅ mini2 setup complete!"
echo ""
echo "Environment:"
echo "  Python: $(/Users/mini2/.venv/bin/python --version)"
echo "  MLX: $(/Users/mini2/.venv/bin/python -c 'import mlx; print(mlx.__version__ if hasattr(mlx, "__version__") else "installed")')"
echo "  Path: /Users/mini2/.venv/bin/python"
echo ""
echo "mini2 is ready for distributed inference!"