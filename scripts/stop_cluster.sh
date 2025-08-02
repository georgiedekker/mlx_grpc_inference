#!/bin/bash
# Stop the distributed MLX inference cluster

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Stopping MLX Distributed Inference Cluster${NC}"

# Function to stop a component
stop_component() {
    local name=$1
    local pid_file="$LOG_DIR/${name}.pid"
    
    echo -n "Stopping $name... "
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            rm -f "$pid_file"
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${YELLOW}not running${NC}"
            rm -f "$pid_file"
        fi
    else
        echo -e "${YELLOW}no PID file${NC}"
    fi
}

# Stop coordinator
stop_component "coordinator"

# Stop workers on remote machines
echo -n "Stopping worker on mini2... "
ssh mini2.local "pkill -f 'python -m src.worker.worker_server' || true"
echo -e "${GREEN}✓${NC}"

echo -n "Stopping worker on master... "
ssh -i ~/.ssh/mlx_master_key georgedekker@master.local "pkill -f 'python -m src.worker.worker_server' || true"
echo -e "${GREEN}✓${NC}"

echo -e "\n${GREEN}✅ Cluster stopped${NC}"