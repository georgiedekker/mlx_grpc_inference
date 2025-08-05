#!/bin/bash
# Test heterogeneous setup locally on mini1 with multiple processes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_NAME="${MODEL_NAME:-microsoft/phi-2}"

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

# Create log directory
mkdir -p "$LOG_DIR"

# Stop any existing processes
print_status "Stopping any existing processes..."
pkill -f "pytorch_heterogeneous_server.py" 2>/dev/null || true
sleep 2

# Test 1: Single device (baseline)
print_status "Test 1: Running single device inference"
cat > "$SCRIPT_DIR/config/test_single.json" << EOF
{
  "cluster": {
    "name": "single-device-test",
    "devices": ["mini1"]
  },
  "model": {
    "name": "$MODEL_NAME",
    "sharding_strategy": "equal",
    "total_layers": 32
  },
  "devices": {
    "mini1": {
      "hostname": "localhost",
      "memory_gb": 16.0,
      "gpu_cores": 10,
      "gpu_memory_gb": 12.0,
      "bandwidth_gbps": 10.0
    }
  },
  "communication": {
    "master_addr": "localhost",
    "master_port": 29501,
    "backend": "gloo"
  }
}
EOF

print_status "Starting single device worker..."
PYTHONUNBUFFERED=1 uv run python "$SCRIPT_DIR/pytorch_heterogeneous_server.py" \
    --rank 0 \
    --world-size 1 \
    --master-addr localhost \
    --master-port 29501 \
    --model-name "$MODEL_NAME" \
    --config "$SCRIPT_DIR/config/test_single.json" \
    > "$LOG_DIR/test_single.log" 2>&1 &

SINGLE_PID=$!
sleep 10

# Check if it started successfully
if ps -p $SINGLE_PID > /dev/null; then
    print_status "✓ Single device test successful"
    # Check layer assignment
    grep -E "(layers|assignment)" "$LOG_DIR/test_single.log" | tail -5
else
    print_error "✗ Single device test failed"
    cat "$LOG_DIR/test_single.log"
fi

kill $SINGLE_PID 2>/dev/null || true
sleep 2

# Test 2: Simulated multi-device on same machine
print_status "\nTest 2: Running simulated heterogeneous setup (2 processes on mini1)"
cat > "$SCRIPT_DIR/config/test_hetero_local.json" << EOF
{
  "cluster": {
    "name": "local-heterogeneous-test",
    "devices": ["worker1", "worker2"]
  },
  "model": {
    "name": "$MODEL_NAME",
    "sharding_strategy": "capability_based",
    "total_layers": 32
  },
  "devices": {
    "worker1": {
      "hostname": "localhost",
      "memory_gb": 8.0,
      "gpu_cores": 5,
      "gpu_memory_gb": 6.0,
      "bandwidth_gbps": 100.0
    },
    "worker2": {
      "hostname": "localhost",
      "memory_gb": 8.0,
      "gpu_cores": 5,
      "gpu_memory_gb": 6.0,
      "bandwidth_gbps": 100.0
    }
  },
  "communication": {
    "master_addr": "localhost",
    "master_port": 29502,
    "backend": "gloo"
  }
}
EOF

# Start worker 1 (master)
print_status "Starting worker 1 (master)..."
PYTHONUNBUFFERED=1 uv run python "$SCRIPT_DIR/pytorch_heterogeneous_server.py" \
    --rank 0 \
    --world-size 2 \
    --master-addr localhost \
    --master-port 29502 \
    --model-name "$MODEL_NAME" \
    --config "$SCRIPT_DIR/config/test_hetero_local.json" \
    > "$LOG_DIR/test_worker1.log" 2>&1 &

WORKER1_PID=$!
sleep 5

# Start worker 2
print_status "Starting worker 2..."
PYTHONUNBUFFERED=1 uv run python "$SCRIPT_DIR/pytorch_heterogeneous_server.py" \
    --rank 1 \
    --world-size 2 \
    --master-addr localhost \
    --master-port 29502 \
    --model-name "$MODEL_NAME" \
    --config "$SCRIPT_DIR/config/test_hetero_local.json" \
    > "$LOG_DIR/test_worker2.log" 2>&1 &

WORKER2_PID=$!
sleep 10

# Check if both started successfully
if ps -p $WORKER1_PID > /dev/null && ps -p $WORKER2_PID > /dev/null; then
    print_status "✓ Multi-process heterogeneous test successful"
    print_status "Worker 1 assignment:"
    grep -E "(assignment:|layers)" "$LOG_DIR/test_worker1.log" | tail -3
    print_status "Worker 2 assignment:"
    grep -E "(assignment:|layers)" "$LOG_DIR/test_worker2.log" | tail -3
else
    print_error "✗ Multi-process test failed"
    print_error "Worker 1 log:"
    tail -20 "$LOG_DIR/test_worker1.log"
    print_error "Worker 2 log:"
    tail -20 "$LOG_DIR/test_worker2.log"
fi

# Cleanup
kill $WORKER1_PID $WORKER2_PID 2>/dev/null || true

print_status "\nTest complete! Check logs in $LOG_DIR/"
print_status "To run the full heterogeneous setup with mini2:"
print_status "1. First run: sudo ./setup_heterogeneous_network.sh"
print_status "2. Set up SSH access to mini2"
print_status "3. Run: CONFIG_FILE=config/heterogeneous_2device.json ./launch_heterogeneous.sh"