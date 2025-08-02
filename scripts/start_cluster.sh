#!/bin/bash
# Start the distributed MLX inference cluster

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting MLX Distributed Inference Cluster${NC}"

# Create logs directory
mkdir -p "$LOG_DIR"

# Check if virtual environment exists
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run: uv venv && uv pip install -e .${NC}"
    exit 1
fi

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Function to check if a process is running
is_running() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    ps -p "$pid" > /dev/null 2>&1
}

# Function to start a component
start_component() {
    local name=$1
    local command=$2
    local log_file="$LOG_DIR/${name}.log"
    
    echo -n "Starting $name... "
    
    # Start the process
    nohup $command > "$log_file" 2>&1 &
    local pid=$!
    
    # Save PID
    echo $pid > "$LOG_DIR/${name}.pid"
    
    # Give it a moment to start
    sleep 2
    
    # Check if it's running
    if is_running $pid; then
        echo -e "${GREEN}✓${NC} (PID: $pid)"
        return 0
    else
        echo -e "${RED}✗${NC}"
        echo "Check logs at: $log_file"
        return 1
    fi
}

# Start worker on mini2
echo -e "\n${YELLOW}Starting worker on mini2...${NC}"
ssh mini2.local "cd /Users/mini2/Movies/mlx_grpc_inference && source .venv/bin/activate && nohup python -m src.worker.worker_server --config config/cluster_config.yaml > logs/worker_mini2.log 2>&1 &"

# Start worker on master
echo -e "\n${YELLOW}Starting worker on master...${NC}"
ssh -i ~/.ssh/mlx_master_key georgedekker@master.local "cd /Users/georgedekker/Movies/mlx_grpc_inference && source .venv/bin/activate && nohup python -m src.worker.worker_server --config config/cluster_config.yaml > logs/worker_master.log 2>&1 &"

# Give workers time to start
echo "Waiting for workers to initialize..."
sleep 5

# Start coordinator API server locally
echo -e "\n${YELLOW}Starting coordinator API server...${NC}"
start_component "coordinator" "python -m src.coordinator.api_server --host 0.0.0.0 --port 8100"

# Wait a bit for everything to stabilize
sleep 3

# Check cluster status
echo -e "\n${YELLOW}Checking cluster status...${NC}"
response=$(curl -s http://localhost:8100/health || echo "Failed")

if [[ "$response" == *"healthy"* ]]; then
    echo -e "${GREEN}✅ Cluster is up and running!${NC}"
    echo
    echo "API endpoint: http://localhost:8100"
    echo "Health check: http://localhost:8100/health"
    echo "Cluster status: http://localhost:8100/cluster/status"
    echo
    echo "Logs are available in: $LOG_DIR"
else
    echo -e "${RED}❌ Cluster failed to start properly${NC}"
    echo "Response: $response"
    exit 1
fi