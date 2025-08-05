#!/bin/bash
# Test the fixed distributed implementation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_NAME="mlx-community/Qwen3-1.7B-8bit"
MINI2_IP="192.168.5.2"
MINI1_IP="192.168.5.1"
MINI2_DIR="/Users/mini2/Movies/mlx_grpc_inference"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Sync files
print_status "Syncing files to mini2..."
ssh mini2@mini2.local "mkdir -p $MINI2_DIR/logs"
rsync -az --exclude='logs' --exclude='__pycache__' --exclude='.venv' \
    "$SCRIPT_DIR"/ mini2@mini2.local:"$MINI2_DIR/"

# Kill any existing processes
print_status "Cleaning up old processes..."
pkill -f pytorch_distributed_fixed.py || true
ssh mini2@mini2.local "pkill -f pytorch_distributed_fixed.py" || true

# Start rank 1 on mini2
print_status "Starting rank 1 on mini2..."
ssh mini2@mini2.local "cd $MINI2_DIR && \
    PYTHONUNBUFFERED=1 \
    RANK=1 \
    WORLD_SIZE=2 \
    MASTER_ADDR=$MINI1_IP \
    MASTER_PORT=29501 \
    MODEL_NAME='$MODEL_NAME' \
    /Users/mini2/.local/bin/uv run python pytorch_distributed_fixed.py \
    > logs/mini2_fixed.log 2>&1 &"

# Start rank 0 on mini1
print_status "Starting rank 0 on mini1..."
PYTHONUNBUFFERED=1 \
RANK=0 \
WORLD_SIZE=2 \
MASTER_ADDR=$MINI1_IP \
MASTER_PORT=29501 \
MODEL_NAME="$MODEL_NAME" \
uv run python "$SCRIPT_DIR/pytorch_distributed_fixed.py" \
    2>&1 | tee "$LOG_DIR/mini1_fixed.log"

# Show mini2 log
print_status "Mini2 log:"
ssh mini2@mini2.local "tail -20 $MINI2_DIR/logs/mini2_fixed.log"