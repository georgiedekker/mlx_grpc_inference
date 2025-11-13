#!/bin/bash

# Wrapper to run with proper uv path on each node
cd ~/Movies/mlx_grpc_inference

if [ -x "/opt/homebrew/bin/uv" ]; then
    # mini1
    /opt/homebrew/bin/uv run python distributed_generation.py
elif [ -x "$HOME/.local/bin/uv" ]; then
    # mini2
    $HOME/.local/bin/uv run python distributed_generation.py
else
    echo "uv not found"
    exit 1
fi