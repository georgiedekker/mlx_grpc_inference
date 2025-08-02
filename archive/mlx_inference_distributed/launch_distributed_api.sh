#!/bin/bash
# Launch script for the distributed MLX API server (coordinator)

set -e

# Default values
CONFIG_FILE="distributed_config.json"
API_PORT=8100
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            API_PORT="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config FILE      Configuration file (default: distributed_config.json)"
            echo "  --port PORT        API server port (default: 8100)"
            echo "  --log-level LEVEL  Log level (default: INFO)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Distributed MLX API Server"
echo "  Config File: $CONFIG_FILE"
echo "  API Port: $API_PORT"
echo "  Log Level: $LOG_LEVEL"

# Set environment variables
export PYTHONUNBUFFERED=1
export LOG_LEVEL=$LOG_LEVEL

# Change to script directory
cd "$(dirname "$0")"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE"
    echo "Creating example configuration..."
    /opt/homebrew/bin/uv run python -c "
from distributed_config_v2 import create_example_config
config = create_example_config()
config.save_json('$CONFIG_FILE')
print('Created example configuration at $CONFIG_FILE')
print('Please edit this file to match your cluster setup')
"
    exit 1
fi

# Check if protocol buffers need to be generated
if [ ! -f "distributed_inference_pb2.py" ] || [ "protos/distributed_inference.proto" -nt "distributed_inference_pb2.py" ]; then
    echo "Generating protocol buffer files..."
    ./generate_proto.sh
fi

# Install required packages if needed
echo "Checking dependencies..."
/opt/homebrew/bin/uv pip list | grep -q grpcio || /opt/homebrew/bin/uv pip install grpcio grpcio-tools protobuf
/opt/homebrew/bin/uv pip list | grep -q fastapi || /opt/homebrew/bin/uv pip install fastapi uvicorn pydantic
/opt/homebrew/bin/uv pip list | grep -q pyyaml || /opt/homebrew/bin/uv pip install pyyaml

# Launch the API server
echo "Launching distributed API server..."
LOG_LEVEL_LOWER=$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')
exec /opt/homebrew/bin/uv run uvicorn distributed_openai_api:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --log-level "$LOG_LEVEL_LOWER"