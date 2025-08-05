#!/bin/bash
# Launch PyTorch Distributed API Server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PyTorch Distributed Inference API Server${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Time: $(date)"

# Create log directory
mkdir -p "$LOG_DIR"

# Configuration
MASTER_IP="192.168.5.1"
MASTER_PORT="12355"
API_PORT="8100"
MODEL_NAME="${MODEL_NAME:-microsoft/phi-2}"

# For testing with smaller model
# MODEL_NAME="microsoft/phi-2"

# For production with larger model
# MODEL_NAME="meta-llama/Llama-2-7b-hf"
# MODEL_NAME="mlx-community/Qwen3-1.7B-8bit"  # Your original model

echo -e "\n${GREEN}Configuration:${NC}"
echo "  Model: $MODEL_NAME"
echo "  Master: $MASTER_IP:$MASTER_PORT"
echo "  API Port: $API_PORT"
echo "  World Size: 2"

# Clean up
echo -e "\n${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "python.*pytorch_api_server" || true
pkill -f "python.*pytorch_distributed_server" || true
lsof -ti :$MASTER_PORT | xargs kill -9 2>/dev/null || true
lsof -ti :$API_PORT | xargs kill -9 2>/dev/null || true
sleep 2

# Copy files to mini2
echo -e "${YELLOW}Copying files to mini2...${NC}"
scp "$SCRIPT_DIR/pytorch_distributed_server.py" 192.168.5.2:~/mlx_grpc_inference/
scp "$SCRIPT_DIR/pytorch_api_server.py" 192.168.5.2:~/mlx_grpc_inference/

# Install transformers on mini2 if needed
echo -e "${YELLOW}Ensuring dependencies on mini2...${NC}"
ssh 192.168.5.2 "pip3 install --user transformers tokenizers accelerate" >/dev/null 2>&1 || true

# Environment setup
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT="$MASTER_PORT"
export MODEL_NAME="$MODEL_NAME"
export WORLD_SIZE="2"

# Start API server (rank 0)
echo -e "\n${GREEN}Starting API Server (Rank 0) on mini1...${NC}"
RANK=0 uv run python pytorch_api_server.py > "$LOG_DIR/api_master_${TIMESTAMP}.log" 2>&1 &
MASTER_PID=$!

echo "API Server PID: $MASTER_PID"

# Wait for master initialization
echo -e "${YELLOW}Waiting for master to initialize...${NC}"
MAX_WAIT=60
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    if grep -q "API server ready\|Uvicorn running" "$LOG_DIR/api_master_${TIMESTAMP}.log" 2>/dev/null; then
        echo -e "${GREEN}✅ Master initialized${NC}"
        break
    fi
    
    if ! kill -0 $MASTER_PID 2>/dev/null; then
        echo -e "${RED}❌ Master process died${NC}"
        echo "Last 20 lines of log:"
        tail -20 "$LOG_DIR/api_master_${TIMESTAMP}.log"
        exit 1
    fi
    
    sleep 2
    WAITED=$((WAITED + 2))
    printf "."
done
echo ""

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}❌ Master failed to initialize after ${MAX_WAIT} seconds${NC}"
    tail -20 "$LOG_DIR/api_master_${TIMESTAMP}.log"
    exit 1
fi

# Additional delay for model loading
sleep 5

# Start worker (rank 1)
echo -e "\n${GREEN}Starting Worker (Rank 1) on mini2...${NC}"
ssh 192.168.5.2 "cd ~/mlx_grpc_inference && \
    RANK=1 WORLD_SIZE=2 MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT \
    MODEL_NAME='$MODEL_NAME' \
    python3 pytorch_api_server.py" > "$LOG_DIR/api_worker_${TIMESTAMP}.log" 2>&1 &
WORKER_PID=$!

echo "Worker PID: $WORKER_PID"

# Wait for both to be ready
echo -e "\n${YELLOW}Waiting for distributed system to be ready...${NC}"
sleep 10

# Test API
echo -e "\n${GREEN}Testing API...${NC}"

# Health check
echo -e "\n${YELLOW}Health check:${NC}"
curl -s http://localhost:$API_PORT/health | python3 -m json.tool || echo "Health check failed"

# Test generation
echo -e "\n${YELLOW}Testing generation:${NC}"
RESPONSE=$(curl -s -X POST http://localhost:$API_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": "Hello! Please count to 5."}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }' 2>/dev/null)

if [ -n "$RESPONSE" ]; then
    echo "$RESPONSE" | python3 -m json.tool
    echo -e "\n${GREEN}✅ API is working!${NC}"
else
    echo -e "${RED}❌ API test failed${NC}"
fi

# Show status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Distributed PyTorch Inference Running${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "API Endpoint: http://localhost:$API_PORT"
echo "Health Check: http://localhost:$API_PORT/health"
echo "Debug Info:   http://localhost:$API_PORT/debug/distributed"
echo ""
echo "Logs:"
echo "  Master: $LOG_DIR/api_master_${TIMESTAMP}.log"
echo "  Worker: $LOG_DIR/api_worker_${TIMESTAMP}.log"
echo ""
echo -e "${YELLOW}To stop:${NC}"
echo "  $0 stop"
echo ""
echo -e "${YELLOW}To monitor:${NC}"
echo "  tail -f $LOG_DIR/api_master_${TIMESTAMP}.log"

# Save PIDs for stop command
echo "$MASTER_PID" > "$LOG_DIR/master.pid"
echo "$WORKER_PID" > "$LOG_DIR/worker.pid"

# Handle stop command
if [ "$1" == "stop" ]; then
    echo -e "\n${YELLOW}Stopping distributed inference...${NC}"
    
    if [ -f "$LOG_DIR/master.pid" ]; then
        kill $(cat "$LOG_DIR/master.pid") 2>/dev/null || true
        rm "$LOG_DIR/master.pid"
    fi
    
    if [ -f "$LOG_DIR/worker.pid" ]; then
        kill $(cat "$LOG_DIR/worker.pid") 2>/dev/null || true
        rm "$LOG_DIR/worker.pid"
    fi
    
    pkill -f "python.*pytorch_api_server" || true
    pkill -f "python.*pytorch_distributed_server" || true
    
    echo -e "${GREEN}✅ Stopped${NC}"
    exit 0
fi