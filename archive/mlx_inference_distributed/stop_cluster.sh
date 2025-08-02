#!/bin/bash
# Stop script for the distributed MLX cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping Distributed MLX Cluster${NC}"

# Change to script directory
cd "$(dirname "$0")"

# Function to stop a local process
stop_local_process() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "Stopping $service_name (PID: $pid)..."
            kill "$pid"
            rm -f "$pid_file"
            echo -e "${GREEN}Stopped $service_name${NC}"
        else
            echo -e "${YELLOW}$service_name not running (stale PID file)${NC}"
            rm -f "$pid_file"
        fi
    else
        echo -e "${YELLOW}No PID file for $service_name${NC}"
    fi
}

# Stop API server
if [ -f "logs/api_server.pid" ]; then
    stop_local_process "logs/api_server.pid" "API Server"
fi

# Stop all gRPC servers
for pid_file in logs/*_server.pid; do
    if [ -f "$pid_file" ]; then
        device_id=$(basename "$pid_file" _server.pid)
        stop_local_process "$pid_file" "gRPC Server ($device_id)"
    fi
done

# Parse configuration to stop remote servers
if [ -f "distributed_config.json" ]; then
    echo -e "\n${YELLOW}Checking for remote servers...${NC}"
    
    REMOTE_DEVICES=$(/opt/homebrew/bin/uv run python -c "
import json
import socket
with open('distributed_config.json') as f:
    config = json.load(f)
    local_hostname = socket.gethostname()
    local_hostname_short = local_hostname.split('.')[0]
    
    for device in config['devices']:
        if device.get('enabled', True):
            hostname = device['hostname']
            # Skip local hosts
            if hostname in ['localhost', '127.0.0.1'] or \
               hostname.startswith(local_hostname) or \
               hostname.startswith(local_hostname_short):
                continue
            print(f\"{device['device_id']}|{hostname}\")
")
    
    # SSH options
    SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"
    
    # Stop remote servers
    while IFS='|' read -r device_id hostname; do
        if [ -n "$hostname" ]; then
            echo -e "Stopping remote server on $device_id ($hostname)..."
            ssh $SSH_OPTS "$hostname" "cd ~/Movies/mlx_distributed && [ -f logs/${device_id}_server.pid ] && kill \$(cat logs/${device_id}_server.pid) && rm -f logs/${device_id}_server.pid || true" 2>/dev/null || true
            echo -e "${GREEN}Stopped remote server on $device_id${NC}"
        fi
    done <<< "$REMOTE_DEVICES"
fi

# Kill any remaining MLX processes
echo -e "\n${YELLOW}Cleaning up any remaining processes...${NC}"
pkill -f "grpc_server.py" || true
pkill -f "distributed_openai_api" || true

echo -e "\n${GREEN}Cluster stopped successfully!${NC}"