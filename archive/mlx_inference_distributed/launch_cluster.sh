#!/bin/bash
# Launch script for the entire distributed MLX cluster

set -e

# Default values
CONFIG_FILE="distributed_config.json"
LAUNCH_WORKERS=true
LAUNCH_API=true
SSH_KEY=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --no-workers)
            LAUNCH_WORKERS=false
            shift
            ;;
        --no-api)
            LAUNCH_API=false
            shift
            ;;
        --ssh-key)
            SSH_KEY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config FILE    Configuration file (default: distributed_config.json)"
            echo "  --no-workers     Don't launch worker nodes (coordinator only)"
            echo "  --no-api         Don't launch API server"
            echo "  --ssh-key FILE   SSH key file for remote connections"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Starting Distributed MLX Cluster${NC}"
echo "  Config File: $CONFIG_FILE"

# Change to script directory
cd "$(dirname "$0")"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Parse configuration to get device information
echo -e "${YELLOW}Parsing configuration...${NC}"
DEVICES=$(/opt/homebrew/bin/uv run python -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
    for device in config['devices']:
        if device.get('enabled', True):
            print(f\"{device['device_id']}|{device['hostname']}|{device['port']}|{device.get('role', 'worker')}\")
")

# SSH options
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
if [ -n "$SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

# Function to check if a host is local
is_local_host() {
    local hostname=$1
    if [[ "$hostname" == "localhost" ]] || [[ "$hostname" == "127.0.0.1" ]]; then
        return 0
    fi
    
    # Check if hostname matches local machine
    local local_hostname=$(hostname)
    local local_hostname_short=$(hostname -s)
    if [[ "$hostname" == "$local_hostname"* ]] || [[ "$hostname" == "$local_hostname_short"* ]]; then
        return 0
    fi
    
    return 1
}

# Function to launch gRPC server on a device
launch_grpc_server() {
    local device_id=$1
    local hostname=$2
    local port=$3
    local role=$4
    
    echo -e "${YELLOW}Launching gRPC server on $device_id ($hostname:$port)...${NC}"
    
    if is_local_host "$hostname"; then
        # Local launch
        ./launch_grpc_server.sh --device-id "$device_id" --port "$port" > "logs/${device_id}_server.log" 2>&1 &
        local pid=$!
        echo "$pid" > "logs/${device_id}_server.pid"
        echo -e "${GREEN}Started local gRPC server on $device_id (PID: $pid)${NC}"
    else
        # Remote launch via SSH
        # First, sync the code to remote machine
        echo "Syncing code to $hostname..."
        ssh $SSH_OPTS "$hostname" "mkdir -p ~/Movies/mlx_distributed"
        rsync -avz --exclude 'logs' --exclude '__pycache__' --exclude '*.pyc' \
            ./ "$hostname:~/Movies/mlx_distributed/"
        
        # Launch server remotely
        ssh $SSH_OPTS "$hostname" "cd ~/Movies/mlx_distributed && nohup ./launch_grpc_server.sh --device-id '$device_id' --port '$port' > 'logs/${device_id}_server.log' 2>&1 &"
        
        echo -e "${GREEN}Started remote gRPC server on $device_id${NC}"
    fi
}

# Create logs directory
mkdir -p logs

# Launch worker nodes
if [ "$LAUNCH_WORKERS" = true ]; then
    echo -e "\n${YELLOW}Launching worker nodes...${NC}"
    
    while IFS='|' read -r device_id hostname port role; do
        if [[ "$role" != "coordinator" ]]; then
            launch_grpc_server "$device_id" "$hostname" "$port" "$role"
            sleep 2  # Give server time to start
        fi
    done <<< "$DEVICES"
fi

# Find coordinator device
COORDINATOR_DEVICE=""
COORDINATOR_HOSTNAME=""
COORDINATOR_PORT=""

while IFS='|' read -r device_id hostname port role; do
    if [[ "$role" == "coordinator" ]]; then
        COORDINATOR_DEVICE="$device_id"
        COORDINATOR_HOSTNAME="$hostname"
        COORDINATOR_PORT="$port"
        break
    fi
done <<< "$DEVICES"

# Launch coordinator gRPC server
if [ -n "$COORDINATOR_DEVICE" ]; then
    echo -e "\n${YELLOW}Launching coordinator node...${NC}"
    launch_grpc_server "$COORDINATOR_DEVICE" "$COORDINATOR_HOSTNAME" "$COORDINATOR_PORT" "coordinator"
    sleep 3  # Give coordinator more time to start
fi

# Launch API server on coordinator
if [ "$LAUNCH_API" = true ] && [ -n "$COORDINATOR_HOSTNAME" ]; then
    echo -e "\n${YELLOW}Launching API server on coordinator...${NC}"
    
    if is_local_host "$COORDINATOR_HOSTNAME"; then
        # Local launch
        ./launch_distributed_api.sh --config "$CONFIG_FILE" > "logs/api_server.log" 2>&1 &
        pid=$!
        echo "$pid" > "logs/api_server.pid"
        echo -e "${GREEN}Started API server (PID: $pid)${NC}"
    else
        # Remote launch
        ssh $SSH_OPTS "$COORDINATOR_HOSTNAME" "cd ~/Movies/mlx_distributed && nohup ./launch_distributed_api.sh --config '$CONFIG_FILE' > 'logs/api_server.log' 2>&1 &"
        echo -e "${GREEN}Started remote API server on $COORDINATOR_HOSTNAME${NC}"
    fi
fi

echo -e "\n${GREEN}Cluster launch complete!${NC}"
echo ""
echo "To check status:"
echo "  - API Health: curl http://localhost:8100/health"
echo "  - Cluster Status: curl http://localhost:8100/cluster/status"
echo "  - Logs: tail -f logs/*.log"
echo ""
echo "To stop the cluster:"
echo "  ./stop_cluster.sh"