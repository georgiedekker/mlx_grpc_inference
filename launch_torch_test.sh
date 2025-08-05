#!/bin/bash
# Launch script for PyTorch distributed test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] PyTorch Distributed Test${NC}"

# Create log directory
mkdir -p "$LOG_DIR"

# Clean up any existing processes
echo "Cleaning up existing processes..."
pkill -f "python.*test_torch_distributed.py" || true
sleep 2

# Copy test script to mini2
echo "Copying test script to mini2..."
scp "$SCRIPT_DIR/test_torch_distributed.py" 192.168.5.2:~/mlx_grpc_inference/

# Environment variables
export MASTER_ADDR="192.168.5.1"
export MASTER_PORT="12355"
export WORLD_SIZE="2"

# Start rank 1 on mini2
echo -e "\n${GREEN}Starting rank 1 on mini2...${NC}"
ssh 192.168.5.2 "cd ~/mlx_grpc_inference && \
    RANK=1 WORLD_SIZE=2 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
    python3 test_torch_distributed.py" > "$LOG_DIR/torch_rank1_${TIMESTAMP}.log" 2>&1 &
RANK1_PID=$!

# Give rank 1 a moment to start
sleep 2

# Start rank 0 on mini1
echo -e "${GREEN}Starting rank 0 on mini1...${NC}"
RANK=0 uv run python test_torch_distributed.py > "$LOG_DIR/torch_rank0_${TIMESTAMP}.log" 2>&1 &
RANK0_PID=$!

# Wait for both processes
echo -e "\n${GREEN}Waiting for processes to complete...${NC}"

# Monitor logs
tail -f "$LOG_DIR/torch_rank0_${TIMESTAMP}.log" &
TAIL_PID=$!

# Wait for rank 0 to complete
wait $RANK0_PID
RANK0_EXIT=$?

# Kill tail
kill $TAIL_PID 2>/dev/null || true

# Wait for rank 1
wait $RANK1_PID
RANK1_EXIT=$?

echo -e "\n${GREEN}=== Results ===${NC}"
echo -e "\n${GREEN}Rank 0 log:${NC}"
cat "$LOG_DIR/torch_rank0_${TIMESTAMP}.log"

echo -e "\n${GREEN}Rank 1 log:${NC}"
cat "$LOG_DIR/torch_rank1_${TIMESTAMP}.log"

if [ $RANK0_EXIT -eq 0 ] && [ $RANK1_EXIT -eq 0 ]; then
    echo -e "\n${GREEN}✅ Distributed test passed!${NC}"
else
    echo -e "\n${RED}❌ Distributed test failed!${NC}"
    exit 1
fi