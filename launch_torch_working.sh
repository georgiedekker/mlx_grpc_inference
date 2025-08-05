#!/bin/bash
# Working launch script for PyTorch distributed across Mac minis

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}PyTorch Distributed - Cross-Machine Test${NC}"
echo -e "${GREEN}Time: $(date)${NC}\n"

# Create log directory
mkdir -p "$LOG_DIR"

# Configuration
MASTER_IP="192.168.5.1"
MASTER_PORT="12355"
TEST_SCRIPT="${1:-test_torch_simple_model.py}"

# Clean up
echo -e "${YELLOW}Cleaning up...${NC}"
pkill -f "python.*torch.*model" || true
lsof -ti :$MASTER_PORT | xargs kill -9 2>/dev/null || true
sleep 2

# Copy files to mini2
echo -e "${YELLOW}Copying files to mini2...${NC}"
scp "$SCRIPT_DIR/$TEST_SCRIPT" 192.168.5.2:~/mlx_grpc_inference/
scp "$SCRIPT_DIR/pytorch_distributed_server.py" 192.168.5.2:~/mlx_grpc_inference/ 2>/dev/null || true

# Environment setup
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT="$MASTER_PORT"
export WORLD_SIZE="2"
export LOCAL_TEST="0"

# For larger models
export MODEL_NAME="${MODEL_NAME:-microsoft/phi-2}"

echo -e "\n${GREEN}Configuration:${NC}"
echo "  Master: $MASTER_IP:$MASTER_PORT"
echo "  World Size: 2"
echo "  Model: $MODEL_NAME"
echo "  Script: $TEST_SCRIPT"

# Start master (rank 0)
echo -e "\n${GREEN}Starting Rank 0 (Master) on mini1...${NC}"
RANK=0 uv run python "$SCRIPT_DIR/$TEST_SCRIPT" > "$LOG_DIR/master_${TIMESTAMP}.log" 2>&1 &
MASTER_PID=$!

echo "Master PID: $MASTER_PID"

# Wait for master to initialize
echo -e "${YELLOW}Waiting for master to initialize...${NC}"
for i in {1..30}; do
    if grep -q "Distributed initialized\|Ready to serve" "$LOG_DIR/master_${TIMESTAMP}.log" 2>/dev/null; then
        echo -e "${GREEN}✅ Master initialized${NC}"
        break
    fi
    
    if ! kill -0 $MASTER_PID 2>/dev/null; then
        echo -e "${RED}❌ Master process died${NC}"
        cat "$LOG_DIR/master_${TIMESTAMP}.log"
        exit 1
    fi
    
    sleep 1
    printf "."
done
echo ""

# Additional delay
sleep 3

# Start worker (rank 1)
echo -e "\n${GREEN}Starting Rank 1 (Worker) on mini2...${NC}"
ssh 192.168.5.2 "cd ~/mlx_grpc_inference && \
    RANK=1 WORLD_SIZE=2 MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT \
    MODEL_NAME='$MODEL_NAME' LOCAL_TEST=0 \
    python3 $TEST_SCRIPT" > "$LOG_DIR/worker_${TIMESTAMP}.log" 2>&1 &
WORKER_PID=$!

echo "Worker PID: $WORKER_PID"

# Monitor
echo -e "\n${GREEN}Monitoring progress...${NC}"
echo -e "${YELLOW}(Following master log, press Ctrl+C to stop monitoring)${NC}\n"

# Function to check both processes
check_processes() {
    if ! kill -0 $MASTER_PID 2>/dev/null; then
        return 1
    fi
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        wait $WORKER_PID
        return $?
    fi
    return 0
}

# Monitor with timeout
MONITOR_TIME=0
MAX_MONITOR=300  # 5 minutes

while check_processes && [ $MONITOR_TIME -lt $MAX_MONITOR ]; do
    # Show last few lines of master log
    tail -n 5 "$LOG_DIR/master_${TIMESTAMP}.log" 2>/dev/null | sed 's/^/[MASTER] /'
    
    # Check for completion
    if grep -q "Test passed\|Shutting down gracefully" "$LOG_DIR/master_${TIMESTAMP}.log" 2>/dev/null; then
        echo -e "\n${GREEN}Master completed successfully${NC}"
        break
    fi
    
    sleep 2
    MONITOR_TIME=$((MONITOR_TIME + 2))
done

# Wait for completion
wait $MASTER_PID
MASTER_EXIT=$?

wait $WORKER_PID
WORKER_EXIT=$?

# Show results
echo -e "\n${GREEN}${'='*80}${NC}"
echo -e "${GREEN}FINAL RESULTS${NC}"
echo -e "${GREEN}${'='*80}${NC}"

echo -e "\n${GREEN}Master Log:${NC}"
tail -n 30 "$LOG_DIR/master_${TIMESTAMP}.log"

echo -e "\n${GREEN}Worker Log:${NC}"
tail -n 30 "$LOG_DIR/worker_${TIMESTAMP}.log"

# Status
echo -e "\n${GREEN}${'='*80}${NC}"
if [ $MASTER_EXIT -eq 0 ] && [ $WORKER_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ SUCCESS: Distributed PyTorch is working across your Mac minis!${NC}"
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo "1. Test with larger model: MODEL_NAME='meta-llama/Llama-2-7b-hf' ./launch_torch_working.sh pytorch_distributed_server.py"
    echo "2. Implement API server on top"
    echo "3. Optimize performance"
else
    echo -e "${RED}❌ FAILURE${NC}"
    echo "Master exit: $MASTER_EXIT, Worker exit: $WORKER_EXIT"
fi

echo -e "\n${YELLOW}Logs saved to:${NC}"
echo "  Master: $LOG_DIR/master_${TIMESTAMP}.log"
echo "  Worker: $LOG_DIR/worker_${TIMESTAMP}.log"