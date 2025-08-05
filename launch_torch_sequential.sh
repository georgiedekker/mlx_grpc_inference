#!/bin/bash
# Sequential launch script for PyTorch distributed with proper timing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] PyTorch Distributed Test - Sequential Launch${NC}"

# Create log directory
mkdir -p "$LOG_DIR"

# Configuration
MASTER_ADDR="192.168.5.1"
MASTER_PORT="12355"

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -i :$port >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Clean up any existing processes
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "python.*test_torch_distributed" || true
sleep 2

# Ensure port is free
if check_port $MASTER_PORT; then
    echo -e "${YELLOW}Port $MASTER_PORT is in use, killing process...${NC}"
    lsof -ti :$MASTER_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Copy test script to mini2
echo "Copying test script to mini2..."
scp "$SCRIPT_DIR/test_torch_distributed_fixed.py" 192.168.5.2:~/mlx_grpc_inference/

# Environment setup
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export WORLD_SIZE="2"

echo -e "\n${GREEN}Configuration:${NC}"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  World Size: $WORLD_SIZE"
echo ""

# Start Rank 0 (Master) first
echo -e "${GREEN}[Step 1/3] Starting Rank 0 (Master) on mini1...${NC}"
export RANK=0
uv run python test_torch_distributed_fixed.py > "$LOG_DIR/torch_rank0_seq_${TIMESTAMP}.log" 2>&1 &
RANK0_PID=$!

echo "Rank 0 PID: $RANK0_PID"
echo "Waiting for master to initialize..."

# Wait for master to bind to port
MAX_WAIT=30
WAITED=0
while ! check_port $MASTER_PORT && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 1
    WAITED=$((WAITED + 1))
    echo -n "."
done
echo ""

if ! check_port $MASTER_PORT; then
    echo -e "${RED}❌ Master failed to bind to port $MASTER_PORT${NC}"
    cat "$LOG_DIR/torch_rank0_seq_${TIMESTAMP}.log"
    exit 1
fi

echo -e "${GREEN}✅ Master is listening on port $MASTER_PORT${NC}"

# Additional delay to ensure master is fully ready
sleep 3

# Start Rank 1 (Worker) on mini2
echo -e "\n${GREEN}[Step 2/3] Starting Rank 1 (Worker) on mini2...${NC}"
ssh 192.168.5.2 "cd ~/mlx_grpc_inference && \
    RANK=1 WORLD_SIZE=2 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
    python3 test_torch_distributed_fixed.py" > "$LOG_DIR/torch_rank1_seq_${TIMESTAMP}.log" 2>&1 &
RANK1_PID=$!

echo "Rank 1 PID: $RANK1_PID"

# Monitor progress
echo -e "\n${GREEN}[Step 3/3] Monitoring progress...${NC}"

# Show rank 0 output
tail -f "$LOG_DIR/torch_rank0_seq_${TIMESTAMP}.log" &
TAIL_PID=$!

# Wait for processes to complete
wait $RANK0_PID
RANK0_EXIT=$?

# Stop tail
kill $TAIL_PID 2>/dev/null || true

wait $RANK1_PID
RANK1_EXIT=$?

# Display results
echo -e "\n${GREEN}${'='*60}${NC}"
echo -e "${GREEN}RESULTS${NC}"
echo -e "${GREEN}${'='*60}${NC}"

echo -e "\n${GREEN}Rank 0 (Master) Output:${NC}"
echo -e "${YELLOW}------------------------${NC}"
cat "$LOG_DIR/torch_rank0_seq_${TIMESTAMP}.log"

echo -e "\n${GREEN}Rank 1 (Worker) Output:${NC}"
echo -e "${YELLOW}------------------------${NC}"
cat "$LOG_DIR/torch_rank1_seq_${TIMESTAMP}.log"

# Check results
if [ $RANK0_EXIT -eq 0 ] && [ $RANK1_EXIT -eq 0 ]; then
    echo -e "\n${GREEN}✅ SUCCESS: Distributed communication test passed!${NC}"
    echo -e "${GREEN}Both processes completed successfully.${NC}"
else
    echo -e "\n${RED}❌ FAILURE: Distributed test failed${NC}"
    echo "Rank 0 exit code: $RANK0_EXIT"
    echo "Rank 1 exit code: $RANK1_EXIT"
    exit 1
fi

echo -e "\n${GREEN}Logs saved to:${NC}"
echo "  - $LOG_DIR/torch_rank0_seq_${TIMESTAMP}.log"
echo "  - $LOG_DIR/torch_rank1_seq_${TIMESTAMP}.log"