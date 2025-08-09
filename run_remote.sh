#!/bin/bash
cd /Users/mini2/Movies/mlx_grpc_inference
source .venv/bin/activate 2>/dev/null || true
exec python server_moe.py