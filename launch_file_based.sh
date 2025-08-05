#!/bin/bash
# Launch script for file-based PyTorch distributed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}PyTorch Distributed - File-based Method${NC}"

# Create log directory
mkdir -p "$LOG_DIR"

# Clean up
pkill -f "python.*torch.*test" || true
sleep 2

# Shared file location - both machines need access
# For now, we'll copy the file after creation
COORD_FILE="/tmp/torch_distributed_${TIMESTAMP}"
export TORCH_DIST_FILE="$COORD_FILE"

# Copy test script
echo "Copying test script to mini2..."
scp "$SCRIPT_DIR/file_based_torch_test.py" 192.168.5.2:~/mlx_grpc_inference/

# Start master first
echo -e "\n${GREEN}Starting master (rank 0)...${NC}"
RANK=0 WORLD_SIZE=2 TORCH_DIST_FILE="$COORD_FILE" \
    uv run python file_based_torch_test.py > "$LOG_DIR/file_master_${TIMESTAMP}.log" 2>&1 &
MASTER_PID=$!

# Wait a moment for file creation
sleep 2

# Copy coordination file to mini2
echo "Copying coordination file to mini2..."
scp "$COORD_FILE" 192.168.5.2:"$COORD_FILE"

# Start worker
echo -e "${GREEN}Starting worker (rank 1)...${NC}"
ssh 192.168.5.2 "cd ~/mlx_grpc_inference && \
    RANK=1 WORLD_SIZE=2 TORCH_DIST_FILE='$COORD_FILE' \
    python3 file_based_torch_test.py" > "$LOG_DIR/file_worker_${TIMESTAMP}.log" 2>&1 &
WORKER_PID=$!

# Monitor
echo -e "\n${GREEN}Monitoring...${NC}"
tail -f "$LOG_DIR/file_master_${TIMESTAMP}.log" &
TAIL_PID=$!

# Wait
wait $MASTER_PID
MASTER_EXIT=$?

kill $TAIL_PID 2>/dev/null || true

wait $WORKER_PID
WORKER_EXIT=$?

# Results
echo -e "\n${GREEN}Master Output:${NC}"
cat "$LOG_DIR/file_master_${TIMESTAMP}.log"

echo -e "\n${GREEN}Worker Output:${NC}"
cat "$LOG_DIR/file_worker_${TIMESTAMP}.log"

if [ $MASTER_EXIT -eq 0 ] && [ $WORKER_EXIT -eq 0 ]; then
    echo -e "\n${GREEN}✅ SUCCESS!${NC}"
else
    echo -e "\n${RED}❌ FAILED${NC}"
fi

# Cleanup
rm -f "$COORD_FILE"
ssh 192.168.5.2 "rm -f '$COORD_FILE'" || true