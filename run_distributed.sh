#!/bin/bash

# Run distributed generation with uv environments
# Both nodes use their local uv environments

cd ~/Movies/mlx_grpc_inference
source .venv/bin/activate
python distributed_generation.py