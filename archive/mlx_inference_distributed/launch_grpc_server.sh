#!/bin/bash
# Launch script for gRPC inference server on a device

set -e

# Default values
DEVICE_ID=""
PORT=50051
CONFIG_FILE=""
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
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
            echo "  --device-id ID     Device ID (auto-detected if not provided)"
            echo "  --port PORT        Server port (default: 50051)"
            echo "  --config FILE      Configuration file path"
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

# Auto-detect device ID if not provided
if [ -z "$DEVICE_ID" ]; then
    DEVICE_ID=$(hostname -s)
    echo "Auto-detected device ID: $DEVICE_ID"
fi

echo "Starting gRPC inference server"
echo "  Device ID: $DEVICE_ID"
echo "  Port: $PORT"
echo "  Log Level: $LOG_LEVEL"

# Set environment variables
export PYTHONUNBUFFERED=1
export GRPC_VERBOSITY=$LOG_LEVEL

# Change to script directory
cd "$(dirname "$0")"

# Check if protocol buffers need to be generated
if [ ! -f "distributed_inference_pb2.py" ] || [ "protos/distributed_inference.proto" -nt "distributed_inference_pb2.py" ]; then
    echo "Generating protocol buffer files..."
    ./generate_proto.sh
fi

# Install required packages if needed
echo "Checking dependencies..."
/opt/homebrew/bin/uv pip list | grep -q grpcio || /opt/homebrew/bin/uv pip install grpcio grpcio-tools protobuf

# Launch the server
echo "Launching gRPC server..."
exec /opt/homebrew/bin/uv run python grpc_server.py \
    --device-id "$DEVICE_ID" \
    --port "$PORT"