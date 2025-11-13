#!/bin/bash
# Launch mini1 node (first shard)

echo "Starting mini1 node (layers 0-13)..."

source .venv/bin/activate

python node_server.py \
    --node-id mini1 \
    --host 192.168.5.1 \
    --port 50051 \
    --start-layer 0 \
    --end-layer 13 \
    --next-node 192.168.5.2:50051