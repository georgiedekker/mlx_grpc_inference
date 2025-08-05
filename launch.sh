#!/bin/bash
# MLX Distributed Inference Launcher
# Usage: ./launch.sh [start|stop|restart|status] [model_name]

set -e

# Default configuration
DEFAULT_MODEL="mlx-community/Qwen3-1.7B-8bit"
MODEL=${2:-$DEFAULT_MODEL}
COMMAND=${1:-"start"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Worker configurations
WORKER_NAMES=("mini2" "m4")
WORKER_CONFIGS=(
    "mini2@192.168.5.2:~/Movies/mlx_grpc_inference"
    "georgedekker@192.168.5.3:~/Movies/mlx_grpc_inference"
)

# Files to sync to workers
WORKER_FILES=(
    "worker.py"
    "launch_worker.sh"
    "requirements.txt"
    "pyproject.toml"
    ".python-version"
    "src/__init__.py"
    "src/communication/__init__.py"
    "src/communication/inference_pb2.py"
    "src/communication/inference_pb2_grpc.py"
    "src/communication/tensor_utils.py"
    "protos/inference.proto"
    "protos/generate_protos.sh"
)

# Log function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Kill all MLX processes on a device
kill_processes() {
    local host=$1
    local user_host=$2
    
    if [ "$host" == "local" ]; then
        log "Killing local MLX processes..."
        pkill -f "python.*server.py" 2>/dev/null || true
        pkill -f "python.*worker.py" 2>/dev/null || true
    else
        log "Killing MLX processes on $host..."
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o StrictHostKeyChecking=no $user_host "pkill -f 'python.*server.py' 2>/dev/null || true; pkill -f 'python.*worker.py' 2>/dev/null || true" 2>/dev/null || true
    fi
}

# Sync files to worker
sync_files() {
    local name=$1
    local user_host=$2
    local dest_dir=$3
    
    log "Syncing files to $name..."
    
    # Extract user and host
    local user=$(echo $user_host | cut -d@ -f1)
    local host=$(echo $user_host | cut -d@ -f2 | cut -d: -f1)
    
    # Create directory structure and clean old proto files
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $user@$host "mkdir -p $dest_dir/src/communication $dest_dir/protos && rm -f $dest_dir/src/communication/inference_pb2*.py 2>/dev/null || true" || {
        error "Failed to create directories on $name"
        return 1
    }
    
    # Copy files
    for file in "${WORKER_FILES[@]}"; do
        if [ -f "$file" ]; then
            scp -q -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$file" "$user@$host:$dest_dir/$file" || {
                error "Failed to copy $file to $name"
                return 1
            }
        fi
    done
    
    # Update model in launch script
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $user@$host "cd $dest_dir && sed -i.bak 's|mlx-community/Qwen3-1.7B-8bit|$MODEL|g' worker.py" || true
    
    # Ensure uv is installed via brew and PATH is updated
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $user@$host "export PATH=/opt/homebrew/bin:/usr/local/bin:\$PATH && if ! command -v uv &> /dev/null; then brew install uv; fi" || true
    
    log "✓ Files synced to $name"
    return 0
}

# Start worker on remote device
start_worker() {
    local name=$1
    local user_host=$2
    local dest_dir=$3
    local worker_id=$4
    local total_workers=3
    
    log "Starting worker on $name (worker $worker_id of $total_workers)..."
    
    # Extract user and host
    local user=$(echo $user_host | cut -d@ -f1)
    local host=$(echo $user_host | cut -d@ -f2 | cut -d: -f1)
    
    # Clean and recreate virtual environment
    log "Setting up clean environment on $name..."
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $user@$host "cd $dest_dir && export PATH=/opt/homebrew/bin:/usr/local/bin:\$PATH && rm -rf .venv __pycache__ src/__pycache__ src/communication/__pycache__ *.pyc src/*.pyc src/communication/*.pyc && uv venv .venv && uv pip install -r requirements.txt" || true
    
    # Regenerate proto files fresh
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $user@$host "cd $dest_dir && export PATH=/opt/homebrew/bin:/usr/local/bin:\$PATH && cd protos && ./generate_protos.sh" || true
    
    # Start worker in background and capture output
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $user@$host "cd $dest_dir && export PATH=/opt/homebrew/bin:/usr/local/bin:\$PATH && nohup ./launch_worker.sh $worker_id $total_workers > worker.log 2>&1 &" || {
        error "Failed to start worker on $name"
        return 1
    }
    
    # Wait for worker to be ready (check logs)
    log "Waiting for worker on $name to be ready..."
    local attempts=0
    while [ $attempts -lt 30 ]; do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $user@$host "cd $dest_dir && grep -q 'gRPC server started' worker.log 2>/dev/null"; then
            log "✓ Worker on $name is ready"
            return 0
        fi
        
        # Check for errors
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $user@$host "cd $dest_dir && grep -q 'error\|Error\|ERROR' worker.log 2>/dev/null"; then
            error "Worker on $name encountered an error:"
            ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $user@$host "cd $dest_dir && tail -5 worker.log"
            return 1
        fi
        
        sleep 1
        ((attempts++))
    done
    
    warning "Worker on $name is taking longer than expected to start"
    return 1
}

# Stop all processes
stop_all() {
    log "Stopping all MLX processes..."
    
    # Kill local processes
    kill_processes "local" ""
    
    # Kill worker processes
    for i in ${!WORKER_NAMES[@]}; do
        local name=${WORKER_NAMES[$i]}
        local config=${WORKER_CONFIGS[$i]}
        local user_host=$(echo $config | cut -d: -f1)
        kill_processes $name $user_host
    done
    
    log "All processes stopped"
}

# Start all processes
start_all() {
    log "Starting MLX distributed inference with model: $MODEL"
    
    # First, stop any existing processes
    stop_all
    
    # Sync files to workers
    for i in ${!WORKER_NAMES[@]}; do
        local name=${WORKER_NAMES[$i]}
        local config=${WORKER_CONFIGS[$i]}
        local user_host=$(echo $config | cut -d: -f1)
        local dest_dir=$(echo $config | cut -d: -f2)
        
        sync_files $name $user_host $dest_dir || {
            error "Failed to sync files to $name"
            return 1
        }
    done
    
    # Start workers
    for i in ${!WORKER_NAMES[@]}; do
        local name=${WORKER_NAMES[$i]}
        local config=${WORKER_CONFIGS[$i]}
        local user_host=$(echo $config | cut -d: -f1)
        local dest_dir=$(echo $config | cut -d: -f2)
        local worker_id=$((i + 1))
        
        start_worker $name $user_host $dest_dir $worker_id || {
            error "Failed to start worker on $name"
            return 1
        }
    done
    
    # Update local server.py with the model
    sed -i.bak "s|mlx-community/Qwen3-1.7B-8bit|$MODEL|g" server.py
    
    # Start coordinator/server
    log "Starting coordinator/API server..."
    nohup uv run python server.py > server.log 2>&1 &
    local server_pid=$!
    
    # Wait for server to be ready
    log "Waiting for API server to be ready..."
    local attempts=0
    while [ $attempts -lt 30 ]; do
        if curl -s http://localhost:8100/health > /dev/null 2>&1; then
            log "✓ API server is ready"
            echo ""
            log "=== MLX Distributed Inference System Started ==="
            log "Model: $MODEL"
            log "API Endpoint: http://localhost:8100"
            log "Health Check: http://localhost:8100/health"
            log "Logs: tail -f server.log"
            echo ""
            return 0
        fi
        sleep 1
        ((attempts++))
    done
    
    error "API server failed to start"
    tail -10 server.log
    return 1
}

# Check status
check_status() {
    log "Checking MLX distributed inference status..."
    echo ""
    
    # Check local server
    if pgrep -f "python.*server.py" > /dev/null; then
        echo -e "${GREEN}✓${NC} Coordinator/API server: Running"
        
        # Check API health
        if curl -s http://localhost:8100/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} API endpoint: Healthy"
            curl -s http://localhost:8100/health | jq '.workers' 2>/dev/null || true
        else
            echo -e "${RED}✗${NC} API endpoint: Not responding"
        fi
    else
        echo -e "${RED}✗${NC} Coordinator/API server: Not running"
    fi
    
    echo ""
    
    # Check workers
    for i in ${!WORKER_NAMES[@]}; do
        local name=${WORKER_NAMES[$i]}
        local config=${WORKER_CONFIGS[$i]}
        local user_host=$(echo $config | cut -d: -f1)
        local host=$(echo $user_host | cut -d@ -f2)
        local worker_id=$((i + 1))
        
        # Check if worker process is running
        if ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no $user_host "pgrep -f 'python.*worker.py'" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Worker $worker_id ($name @ $host): Running"
        else
            echo -e "${RED}✗${NC} Worker $worker_id ($name @ $host): Not running"
        fi
    done
    
    echo ""
}

# Test the system
test_inference() {
    log "Testing inference..."
    curl -X POST http://localhost:8100/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 20,
            "temperature": 0.7
        }' | jq
}

# Main command handling
case $COMMAND in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        check_status
        ;;
    test)
        test_inference
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|status|test] [model_name]"
        echo ""
        echo "Commands:"
        echo "  start   - Start all workers and API server"
        echo "  stop    - Stop all workers and API server"
        echo "  restart - Restart all components"
        echo "  status  - Check status of all components"
        echo "  test    - Test inference with a simple query"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 restart mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
        echo "  $0 stop"
        exit 1
        ;;
esac