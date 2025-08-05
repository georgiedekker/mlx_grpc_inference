#!/bin/bash
# Simple distributed PyTorch inference between mini1 and mini2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_NAME="${MODEL_NAME:-mlx-community/Qwen3-1.7B-8bit}"
MINI2_IP="192.168.5.2"
MINI1_IP="192.168.5.1"
MINI2_DIR="/Users/mini2/Movies/mlx_grpc_inference"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

mkdir -p "$LOG_DIR"

# Stop function
stop_services() {
    print_status "Stopping services..."
    
    # Stop local worker
    if [ -f "$LOG_DIR/mini1_worker.pid" ]; then
        kill $(cat "$LOG_DIR/mini1_worker.pid") 2>/dev/null || true
        rm -f "$LOG_DIR/mini1_worker.pid"
    fi
    
    # Stop remote worker
    ssh mini2@mini2.local "pkill -f pytorch_distributed_server.py" 2>/dev/null || true
    
    print_status "All services stopped"
}

# Start function
start_services() {
    print_status "Starting distributed inference between mini1 and mini2"
    print_status "Model: $MODEL_NAME"
    
    # Stop any existing services
    stop_services
    
    # Sync code to mini2
    print_status "Syncing code to mini2..."
    # Create directory on mini2 if it doesn't exist
    ssh mini2@mini2.local "mkdir -p $MINI2_DIR/logs"
    rsync -az --exclude='logs' --exclude='__pycache__' --exclude='.venv' \
        "$SCRIPT_DIR"/ mini2@mini2.local:"$MINI2_DIR/"
    
    # Start mini2 worker (rank 1)
    print_status "Starting worker on mini2 (rank 1)..."
    ssh mini2@mini2.local "cd $MINI2_DIR && \
        PYTHONUNBUFFERED=1 MODEL_NAME='$MODEL_NAME' /Users/mini2/.local/bin/uv run python pytorch_distributed_server.py \
        --rank 1 \
        --world-size 2 \
        --master-addr $MINI1_IP \
        --master-port 29500 \
        --model-name '$MODEL_NAME' \
        > logs/mini2_worker.log 2>&1 & \
        echo \$!"
    
    sleep 3
    
    # Start mini1 worker (rank 0)
    print_status "Starting worker on mini1 (rank 0)..."
    PYTHONUNBUFFERED=1 MODEL_NAME="$MODEL_NAME" uv run python "$SCRIPT_DIR/pytorch_distributed_server.py" \
        --rank 0 \
        --world-size 2 \
        --master-addr $MINI1_IP \
        --master-port 29500 \
        --model-name "$MODEL_NAME" \
        > "$LOG_DIR/mini1_worker.log" 2>&1 &
    
    echo $! > "$LOG_DIR/mini1_worker.pid"
    
    print_status "Waiting for workers to initialize..."
    sleep 15
    
    # Check if workers are running
    if ps -p $(cat "$LOG_DIR/mini1_worker.pid") > /dev/null; then
        print_status "✓ mini1 worker is running"
    else
        print_error "✗ mini1 worker failed"
        tail -20 "$LOG_DIR/mini1_worker.log"
        stop_services
        exit 1
    fi
    
    if ssh mini2@mini2.local "pgrep -f pytorch_distributed_server.py" > /dev/null; then
        print_status "✓ mini2 worker is running"
    else
        print_error "✗ mini2 worker failed"
        ssh mini2@mini2.local "tail -20 $MINI2_DIR/logs/mini2_worker.log"
        stop_services
        exit 1
    fi
    
    # The API server runs as part of rank 0 (mini1)
    print_status "Checking API server on rank 0..."
    sleep 5
    
    # Test API
    if curl -s http://localhost:8100/health > /dev/null; then
        print_status "✓ API server is ready at http://localhost:8100"
    else
        print_error "✗ API server not responding"
        print_error "Check mini1 worker log for API server status"
        tail -20 "$LOG_DIR/mini1_worker.log" | grep -E "(API|ERROR)"
    fi
    
    print_status "Distributed inference is running!"
    print_status "mini1: 16 layers, mini2: 16 layers"
    print_status "API endpoint: http://localhost:8100"
}

# Test function  
test_inference() {
    print_status "Testing inference..."
    
    curl -X POST http://localhost:8100/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_NAME"'",
            "prompt": "The capital of France is",
            "max_tokens": 20,
            "temperature": 0.7
        }' | uv run python -m json.tool
}

# Test basic communication function
test_basic_comm() {
    print_status "Testing basic distributed communication..."
    
    # Stop any existing services
    stop_services
    
    # Sync test files to mini2
    print_status "Syncing test files to mini2..."
    ssh mini2@mini2.local "mkdir -p $MINI2_DIR/logs"
    rsync -az --exclude='logs' --exclude='__pycache__' --exclude='.venv' \
        "$SCRIPT_DIR"/ mini2@mini2.local:"$MINI2_DIR/"
    
    # Test on mini2 first
    print_status "Starting test worker on mini2..."
    ssh mini2@mini2.local "cd $MINI2_DIR && \
        RANK=1 WORLD_SIZE=2 MASTER_ADDR=$MINI1_IP MASTER_PORT=29500 \
        /Users/mini2/.local/bin/uv run python test_distributed_step_by_step.py \
        > logs/test_mini2.log 2>&1 &"
    
    sleep 2
    
    # Test on mini1
    print_status "Starting test on mini1..."
    RANK=0 WORLD_SIZE=2 MASTER_ADDR=$MINI1_IP MASTER_PORT=29500 \
        uv run python "$SCRIPT_DIR/test_distributed_step_by_step.py" \
        2>&1 | tee "$LOG_DIR/test_mini1.log"
    
    # Show mini2 log
    print_status "Mini2 test log:"
    ssh mini2@mini2.local "cat $MINI2_DIR/logs/test_mini2.log"
}

# Start debug version
start_debug() {
    print_status "Starting DEBUG version with comprehensive logging"
    print_status "Model: $MODEL_NAME"
    
    # Stop any existing services
    stop_services
    
    # Sync code to mini2
    print_status "Syncing code to mini2..."
    ssh mini2@mini2.local "mkdir -p $MINI2_DIR/logs"
    rsync -az --exclude='logs' --exclude='__pycache__' --exclude='.venv' \
        "$SCRIPT_DIR"/ mini2@mini2.local:"$MINI2_DIR/"
    
    # Start mini2 worker (rank 1) - debug version
    print_status "Starting DEBUG worker on mini2 (rank 1)..."
    ssh mini2@mini2.local "cd $MINI2_DIR && \
        PYTHONUNBUFFERED=1 \
        RANK=1 \
        WORLD_SIZE=2 \
        MASTER_ADDR=$MINI1_IP \
        MASTER_PORT=29500 \
        MODEL_NAME='$MODEL_NAME' \
        /Users/mini2/.local/bin/uv run python pytorch_distributed_debug.py \
        > logs/mini2_debug.log 2>&1 & \
        echo \$!"
    
    sleep 5
    
    # Start mini1 worker (rank 0) - debug version
    print_status "Starting DEBUG worker on mini1 (rank 0)..."
    PYTHONUNBUFFERED=1 \
    RANK=0 \
    WORLD_SIZE=2 \
    MASTER_ADDR=$MINI1_IP \
    MASTER_PORT=29500 \
    MODEL_NAME="$MODEL_NAME" \
    uv run python "$SCRIPT_DIR/pytorch_distributed_debug.py" \
        > "$LOG_DIR/mini1_debug.log" 2>&1 &
    
    echo $! > "$LOG_DIR/mini1_debug.pid"
    
    print_status "Debug servers starting... Check logs with:"
    print_status "  $0 logs debug-mini1"
    print_status "  $0 logs debug-mini2"
}

# Main
case "${1:-start}" in
    start)
        start_services
        ;;
    debug)
        start_debug
        ;;
    test-comm)
        test_basic_comm
        ;;
    stop)
        stop_services
        # Also stop debug processes
        if [ -f "$LOG_DIR/mini1_debug.pid" ]; then
            kill $(cat "$LOG_DIR/mini1_debug.pid") 2>/dev/null || true
            rm -f "$LOG_DIR/mini1_debug.pid"
        fi
        ssh mini2@mini2.local "pkill -f pytorch_distributed_debug.py" 2>/dev/null || true
        ;;
    test)
        test_inference
        ;;
    status)
        echo "=== Regular Services ==="
        echo "mini1 worker: $(ps -p $(cat "$LOG_DIR/mini1_worker.pid" 2>/dev/null) > /dev/null 2>&1 && echo "Running" || echo "Stopped")"
        echo "mini2 worker: $(ssh mini2@mini2.local "pgrep -f pytorch_distributed_server.py" > /dev/null 2>&1 && echo "Running" || echo "Stopped")"
        echo "API server: $(curl -s http://localhost:8100/health > /dev/null 2>&1 && echo "Running on port 8100" || echo "Not responding")"
        echo ""
        echo "=== Debug Services ==="
        echo "mini1 debug: $(ps -p $(cat "$LOG_DIR/mini1_debug.pid" 2>/dev/null) > /dev/null 2>&1 && echo "Running" || echo "Stopped")"
        echo "mini2 debug: $(ssh mini2@mini2.local "pgrep -f pytorch_distributed_debug.py" > /dev/null 2>&1 && echo "Running" || echo "Stopped")"
        ;;
    logs)
        case "${2:-mini1}" in
            mini1)
                tail -f "$LOG_DIR/mini1_worker.log"
                ;;
            mini2)
                ssh mini2@mini2.local "tail -f $MINI2_DIR/logs/mini2_worker.log"
                ;;
            debug-mini1)
                tail -f "$LOG_DIR/mini1_debug.log"
                ;;
            debug-mini2)
                ssh mini2@mini2.local "tail -f $MINI2_DIR/logs/mini2_debug.log"
                ;;
            test-mini1)
                cat "$LOG_DIR/test_mini1.log"
                ;;
            test-mini2)
                ssh mini2@mini2.local "cat $MINI2_DIR/logs/test_mini2.log"
                ;;
            api)
                tail -f "$LOG_DIR/api_server.log"
                ;;
        esac
        ;;
    *)
        echo "Usage: $0 {start|stop|debug|test-comm|test|status|logs [mini1|mini2|debug-mini1|debug-mini2|test-mini1|test-mini2|api]}"
        echo ""
        echo "Commands:"
        echo "  start      - Start normal distributed inference"
        echo "  debug      - Start debug version with comprehensive logging"
        echo "  test-comm  - Test basic distributed communication"
        echo "  stop       - Stop all services"
        echo "  test       - Test inference API"
        echo "  status     - Show service status"
        echo "  logs       - Show logs"
        echo ""
        echo "Environment variables:"
        echo "  MODEL_NAME - Model to load (default: mlx-community/Qwen3-1.7B-8bit)"
        exit 1
        ;;
esac