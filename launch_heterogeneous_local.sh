#!/bin/bash
# Launch heterogeneous inference locally on mini1 (no SSH required)
# Simulates multi-device setup using multiple processes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_NAME="${MODEL_NAME:-microsoft/phi-2}"
API_PORT="${API_PORT:-8000}"
NUM_WORKERS="${NUM_WORKERS:-2}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Create log directory
mkdir -p "$LOG_DIR"

# Stop function
stop_services() {
    print_status "Stopping all services..."
    pkill -f "pytorch_heterogeneous_server.py" 2>/dev/null || true
    pkill -f "pytorch_heterogeneous_api.py" 2>/dev/null || true
    rm -f "$LOG_DIR"/*.pid
    print_status "All services stopped"
}

# Create local config
create_local_config() {
    print_status "Creating local heterogeneous configuration..."
    
    # Determine layer distribution based on NUM_WORKERS
    if [ "$NUM_WORKERS" -eq 2 ]; then
        # Simulate mini1 + mini2
        STRATEGY="equal"
        DEVICES='["local-mini1", "local-mini2"]'
        DEVICE_SPECS='{
            "local-mini1": {
                "hostname": "localhost",
                "memory_gb": 8.0,
                "gpu_cores": 5,
                "gpu_memory_gb": 6.0,
                "bandwidth_gbps": 100.0
            },
            "local-mini2": {
                "hostname": "localhost",
                "memory_gb": 8.0,
                "gpu_cores": 5,
                "gpu_memory_gb": 6.0,
                "bandwidth_gbps": 100.0
            }
        }'
    else
        # Simulate mini1 + mini2 + master
        STRATEGY="capability_based"
        DEVICES='["local-mini1", "local-mini2", "local-master"]'
        DEVICE_SPECS='{
            "local-mini1": {
                "hostname": "localhost",
                "memory_gb": 5.0,
                "gpu_cores": 3,
                "gpu_memory_gb": 4.0,
                "bandwidth_gbps": 100.0
            },
            "local-mini2": {
                "hostname": "localhost",
                "memory_gb": 5.0,
                "gpu_cores": 3,
                "gpu_memory_gb": 4.0,
                "bandwidth_gbps": 100.0
            },
            "local-master": {
                "hostname": "localhost",
                "memory_gb": 6.0,
                "gpu_cores": 4,
                "gpu_memory_gb": 4.0,
                "bandwidth_gbps": 100.0
            }
        }'
    fi
    
    cat > "$SCRIPT_DIR/config/heterogeneous_local.json" << EOF
{
  "cluster": {
    "name": "local-heterogeneous-cluster",
    "devices": $DEVICES
  },
  "model": {
    "name": "$MODEL_NAME",
    "sharding_strategy": "$STRATEGY"
  },
  "devices": $DEVICE_SPECS,
  "communication": {
    "master_addr": "localhost",
    "master_port": 29503,
    "backend": "gloo"
  }
}
EOF
    
    print_status "Configuration created for $NUM_WORKERS workers with $STRATEGY strategy"
}

# Launch workers
launch_workers() {
    print_status "Launching $NUM_WORKERS workers locally..."
    
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        print_status "Starting worker $i..."
        
        PYTHONUNBUFFERED=1 uv run python "$SCRIPT_DIR/pytorch_heterogeneous_server.py" \
            --rank $i \
            --world-size $NUM_WORKERS \
            --master-addr localhost \
            --master-port 29503 \
            --model-name "$MODEL_NAME" \
            --config "$SCRIPT_DIR/config/heterogeneous_local.json" \
            > "$LOG_DIR/worker_$i.log" 2>&1 &
        
        echo $! > "$LOG_DIR/worker_$i.pid"
        
        # Give master time to initialize
        if [ $i -eq 0 ]; then
            sleep 5
        else
            sleep 2
        fi
    done
    
    # Wait for all workers to be ready
    print_status "Waiting for workers to initialize..."
    sleep 10
    
    # Check workers
    ALL_READY=true
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        if [ -f "$LOG_DIR/worker_$i.pid" ]; then
            PID=$(cat "$LOG_DIR/worker_$i.pid")
            if ps -p $PID > /dev/null; then
                print_status "✓ Worker $i running (PID: $PID)"
                # Show layer assignment
                grep -E "assignment:|layers" "$LOG_DIR/worker_$i.log" | tail -2
            else
                print_error "✗ Worker $i failed"
                tail -10 "$LOG_DIR/worker_$i.log"
                ALL_READY=false
            fi
        fi
    done
    
    if [ "$ALL_READY" = false ]; then
        print_error "Some workers failed to start"
        stop_services
        exit 1
    fi
}

# Launch API server
launch_api() {
    print_status "Launching API server on port $API_PORT..."
    
    PYTHONUNBUFFERED=1 MODEL_NAME="$MODEL_NAME" \
        CONFIG_FILE="$SCRIPT_DIR/config/heterogeneous_local.json" \
        uv run python "$SCRIPT_DIR/pytorch_heterogeneous_api.py" \
        --port "$API_PORT" \
        > "$LOG_DIR/api_server.log" 2>&1 &
    
    echo $! > "$LOG_DIR/api_server.pid"
    
    # Wait for API to start
    sleep 5
    
    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        print_status "✓ API server ready at http://localhost:$API_PORT"
    else
        print_error "✗ API server failed to start"
        tail -20 "$LOG_DIR/api_server.log"
        stop_services
        exit 1
    fi
}

# Test the setup
test_inference() {
    print_status "Testing inference..."
    
    # Test completion
    RESPONSE=$(curl -s -X POST "http://localhost:$API_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_NAME"'",
            "prompt": "The capital of France is",
            "max_tokens": 20,
            "temperature": 0.7
        }')
    
    if [ -n "$RESPONSE" ]; then
        print_status "✓ Inference test successful"
        echo "$RESPONSE" | uv run python -m json.tool
    else
        print_error "✗ Inference test failed"
    fi
    
    # Show cluster status
    print_status "Cluster status:"
    curl -s "http://localhost:$API_PORT/cluster/status" | uv run python -m json.tool
}

# Main execution
case "${1:-start}" in
    start)
        stop_services
        create_local_config
        launch_workers
        launch_api
        test_inference
        
        print_status "Local heterogeneous cluster running!"
        print_status "API available at: http://localhost:$API_PORT"
        print_status "Logs in: $LOG_DIR"
        print_status "Stop with: $0 stop"
        ;;
        
    stop)
        stop_services
        ;;
        
    status)
        print_status "Checking status..."
        for i in $(seq 0 $((NUM_WORKERS - 1))); do
            if [ -f "$LOG_DIR/worker_$i.pid" ]; then
                PID=$(cat "$LOG_DIR/worker_$i.pid")
                if ps -p $PID > /dev/null; then
                    echo -e "Worker $i: ${GREEN}Running${NC} (PID: $PID)"
                else
                    echo -e "Worker $i: ${RED}Stopped${NC}"
                fi
            fi
        done
        
        if [ -f "$LOG_DIR/api_server.pid" ]; then
            PID=$(cat "$LOG_DIR/api_server.pid")
            if ps -p $PID > /dev/null; then
                echo -e "API Server: ${GREEN}Running${NC} (PID: $PID)"
                curl -s "http://localhost:$API_PORT/cluster/status" | uv run python -m json.tool
            else
                echo -e "API Server: ${RED}Stopped${NC}"
            fi
        fi
        ;;
        
    logs)
        WORKER="${2:-0}"
        tail -f "$LOG_DIR/worker_$WORKER.log"
        ;;
        
    *)
        echo "Usage: $0 {start|stop|status|logs [worker_num]}"
        echo ""
        echo "Environment variables:"
        echo "  MODEL_NAME   - Model to load (default: microsoft/phi-2)"
        echo "  NUM_WORKERS  - Number of workers (default: 2, max: 3)"
        echo "  API_PORT     - API port (default: 8000)"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start with 2 workers"
        echo "  NUM_WORKERS=3 $0 start      # Start with 3 workers (heterogeneous)"
        echo "  $0 logs 1                   # View logs for worker 1"
        exit 1
        ;;
esac