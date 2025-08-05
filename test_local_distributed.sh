#!/bin/bash
# Test PyTorch distributed locally first

echo "Testing PyTorch distributed on localhost..."

# Clean up
pkill -f "python.*test_torch_simple_model" || true
sleep 1

# Test with 2 local processes
export LOCAL_TEST=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Start rank 0
echo "Starting rank 0..."
RANK=0 WORLD_SIZE=2 uv run python test_torch_simple_model.py &
PID0=$!

sleep 2

# Start rank 1
echo "Starting rank 1..."
RANK=1 WORLD_SIZE=2 uv run python test_torch_simple_model.py &
PID1=$!

# Wait for completion
wait $PID0
EXIT0=$?

wait $PID1
EXIT1=$?

if [ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ]; then
    echo -e "\n✅ Local distributed test PASSED!"
else
    echo -e "\n❌ Local distributed test FAILED!"
fi