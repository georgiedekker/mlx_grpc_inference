#!/bin/bash
# Launch mini2 node (second shard)

echo "Starting mini2 node (layers 14-27)..."

source .venv/bin/activate

python node_server.py \
    --node-id mini2 \
    --host 192.168.5.2 \
    --port 50051 \
    --start-layer 14 \
    --end-layer 27