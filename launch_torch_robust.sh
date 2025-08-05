#!/bin/bash
# Robust launch script for PyTorch distributed on macOS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}PyTorch Distributed - Robust Launch${NC}"
echo -e "${GREEN}Time: $(date)${NC}"

# Create log directory
mkdir -p "$LOG_DIR"

# Network interface configuration
export GLOO_SOCKET_IFNAME=bridge0
echo -e "${YELLOW}Using network interface: $GLOO_SOCKET_IFNAME${NC}"

# Master configuration
MASTER_IP="192.168.5.1"
MASTER_PORT="12355"

# Function to check if port is listening
check_port_listening() {
    lsof -i tcp:$1 -sTCP:LISTEN >/dev/null 2>&1
}

# Function to kill process on port
kill_port() {
    local port=$1
    lsof -ti :$port | xargs kill -9 2>/dev/null || true
}

# Clean up
echo -e "\n${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "python.*torch.*test" || true
kill_port $MASTER_PORT
sleep 2

# Test network connectivity first
echo -e "\n${GREEN}Testing network connectivity...${NC}"
echo "Testing connection to mini2 (192.168.5.2)..."
if ping -c 1 -W 1 192.168.5.2 >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Network connectivity OK${NC}"
else
    echo -e "${RED}❌ Cannot reach mini2${NC}"
    exit 1
fi

# Copy test script
echo -e "\n${YELLOW}Copying test script to mini2...${NC}"
scp "$SCRIPT_DIR/simple_torch_test.py" 192.168.5.2:~/mlx_grpc_inference/

# Start master (rank 0)
echo -e "\n${GREEN}[1/4] Starting master (rank 0) on mini1...${NC}"
RANK=0 WORLD_SIZE=2 GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
    uv run python simple_torch_test.py > "$LOG_DIR/torch_master_${TIMESTAMP}.log" 2>&1 &
MASTER_PID=$!

echo "Master PID: $MASTER_PID"

# Wait for master to bind
echo -e "\n${GREEN}[2/4] Waiting for master to bind to port $MASTER_PORT...${NC}"
MAX_WAIT=30
WAITED=0
while ! check_port_listening $MASTER_PORT && [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 $MASTER_PID 2>/dev/null; then
        echo -e "\n${RED}❌ Master process died!${NC}"
        echo "Master log:"
        cat "$LOG_DIR/torch_master_${TIMESTAMP}.log"
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
    printf "."
done
echo ""

if ! check_port_listening $MASTER_PORT; then
    echo -e "${RED}❌ Master failed to bind after ${MAX_WAIT} seconds${NC}"
    echo "Master log:"
    cat "$LOG_DIR/torch_master_${TIMESTAMP}.log"
    exit 1
fi

echo -e "${GREEN}✅ Master is listening on port $MASTER_PORT${NC}"

# Verify with netstat
echo -e "\n${YELLOW}Network status:${NC}"
netstat -an | grep $MASTER_PORT | grep LISTEN || true

# Additional delay for master initialization
sleep 2

# Start worker (rank 1)
echo -e "\n${GREEN}[3/4] Starting worker (rank 1) on mini2...${NC}"
ssh 192.168.5.2 "cd ~/mlx_grpc_inference && \
    RANK=1 WORLD_SIZE=2 GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
    python3 simple_torch_test.py" > "$LOG_DIR/torch_worker_${TIMESTAMP}.log" 2>&1 &
WORKER_PID=$!

echo "Worker PID: $WORKER_PID"

# Monitor both processes
echo -e "\n${GREEN}[4/4] Monitoring processes...${NC}"

# Show master output in real-time
tail -f "$LOG_DIR/torch_master_${TIMESTAMP}.log" &
TAIL_PID=$!

# Wait for processes
wait $MASTER_PID
MASTER_EXIT=$?

kill $TAIL_PID 2>/dev/null || true

wait $WORKER_PID
WORKER_EXIT=$?

# Display results
echo -e "\n${GREEN}${'='*80}${NC}"
echo -e "${GREEN}RESULTS${NC}"
echo -e "${GREEN}${'='*80}${NC}"

echo -e "\n${GREEN}Master (Rank 0) Output:${NC}"
cat "$LOG_DIR/torch_master_${TIMESTAMP}.log"

echo -e "\n${GREEN}Worker (Rank 1) Output:${NC}"
cat "$LOG_DIR/torch_worker_${TIMESTAMP}.log"

# Final status
echo -e "\n${GREEN}${'='*80}${NC}"
if [ $MASTER_EXIT -eq 0 ] && [ $WORKER_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ SUCCESS: Distributed test passed!${NC}"
    echo -e "${GREEN}PyTorch distributed is working correctly on your Mac minis!${NC}"
else
    echo -e "${RED}❌ FAILURE: Distributed test failed${NC}"
    echo "Master exit code: $MASTER_EXIT"
    echo "Worker exit code: $WORKER_EXIT"
    exit 1
fi

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Implement model sharding in pytorch_distributed_server.py"
echo "2. Test with actual model inference"
echo "3. Build API server on top"