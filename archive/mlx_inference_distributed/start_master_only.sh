#!/bin/bash
# Start just the master node for testing

echo "🚀 Starting master node only for testing..."

# Kill any existing processes
echo "🛑 Cleaning up existing processes..."
pkill -f "python.*distributed_api.py" || true

# Create logs directory if it doesn't exist
mkdir -p logs

# Start master on mini1 with single-node configuration
echo "🟢 Starting master API server on mini1..."
cd /Users/mini1/Movies/mlx_inference_distributed
source .venv/bin/activate

# Set environment for single node operation
export GRPC_DNS_RESOLVER=native
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export DISTRIBUTED_CONFIG=distributed_config.json
export API_PORT=8100

# Start the API server
python distributed_api.py 2>&1 | tee logs/api_server_single.log