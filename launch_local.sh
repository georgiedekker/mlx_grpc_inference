#!/bin/bash
# Test locally on single device first

# Kill any existing processes
pkill -f "python.*server.py"

# Set environment for single device
export MLX_RANK=0
export MLX_WORLD_SIZE=1

echo "Launching single-device inference..."
uv run python server.py