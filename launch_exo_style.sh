#!/bin/bash

# Launch distributed inference using exo's architecture
# Each node runs independently, no MPI

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting MLX Distributed Inference (Exo-style)${NC}"
echo "================================================"

# Kill any existing processes
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
pkill -f exo_node.py 2>/dev/null || true
pkill -f exo_coordinator.py 2>/dev/null || true
ssh mini2@192.168.5.2 "pkill -f exo_node.py 2>/dev/null || true"
sleep 2

# Copy files to mini2
echo -e "${YELLOW}Copying files to mini2...${NC}"
scp -q exo_node.py mini2@192.168.5.2:/Users/mini2/

# Install aiohttp if needed
echo -e "${YELLOW}Ensuring dependencies...${NC}"
source .venv/bin/activate
python -c "import aiohttp" 2>/dev/null || uv pip install aiohttp
ssh mini2@192.168.5.2 "source /Users/mini2/.venv/bin/activate && python -c 'import aiohttp' 2>/dev/null || /Users/mini2/.local/bin/uv pip install aiohttp"

# Start mini2 node (second shard)
echo -e "${YELLOW}Starting mini2 node (layers 14-27)...${NC}"
ssh mini2@192.168.5.2 "cd /Users/mini2 && source .venv/bin/activate && nohup python exo_node.py \
    --node-id mini2 \
    --host 0.0.0.0 \
    --port 50051 \
    --start-layer 14 \
    --end-layer 27 \
    > mini2_node.log 2>&1 &"

# Wait for mini2 to start
echo -e "${YELLOW}Waiting for mini2 to initialize...${NC}"
sleep 10

# Start mini1 node (first shard)
echo -e "${YELLOW}Starting mini1 node (layers 0-13)...${NC}"
nohup python exo_node.py \
    --node-id mini1 \
    --host 0.0.0.0 \
    --port 50051 \
    --start-layer 0 \
    --end-layer 13 \
    --next-node 192.168.5.2:50051 \
    > mini1_node.log 2>&1 &

# Wait for mini1 to start
echo -e "${YELLOW}Waiting for mini1 to initialize...${NC}"
sleep 10

# Start coordinator
echo -e "${YELLOW}Starting coordinator API server...${NC}"
nohup python exo_coordinator.py > coordinator.log 2>&1 &

# Wait for coordinator
sleep 5

# Check health
echo -e "${YELLOW}Checking system health...${NC}"
if curl -s http://localhost:8100/health | jq '.' 2>/dev/null; then
    echo -e "${GREEN}✅ System is healthy!${NC}"
else
    echo -e "${RED}⚠️ Health check failed${NC}"
fi

echo ""
echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}MLX Distributed Inference Running!${NC}"
echo -e "${GREEN}===================================${NC}"
echo ""
echo "API: http://localhost:8100"
echo "Logs:"
echo "  mini1: tail -f mini1_node.log"
echo "  mini2: ssh mini2@192.168.5.2 'tail -f mini2_node.log'"
echo "  coordinator: tail -f coordinator.log"
echo ""
echo "Test with:"
echo 'curl -X POST http://localhost:8100/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 20}'"'"

echo ""
echo "Stop with: pkill -f exo_node && pkill -f exo_coordinator && ssh mini2@192.168.5.2 'pkill -f exo_node'"