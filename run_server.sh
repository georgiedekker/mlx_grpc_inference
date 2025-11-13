#!/bin/bash
# Wrapper script to run server with correct path on each machine

if [ "$(hostname)" = "mini1.local" ]; then
    cd /Users/mini1/Movies/mlx_grpc_inference
    /opt/homebrew/bin/python3 server_moe.py
else
    cd /Users/mini2/Movies/mlx_grpc_inference
    /Library/Developer/CommandLineTools/usr/bin/python3 server_moe.py
fi