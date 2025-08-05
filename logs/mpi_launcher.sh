#!/bin/bash
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"mini1"* ]] || [[ "$HOSTNAME" == "192.168.5.1" ]]; then
    exec uv run python /Users/mini1/Movies/mlx_grpc_inference/server.py
else
    exec uv run python ~/mlx_grpc_inference/server.py
fi
