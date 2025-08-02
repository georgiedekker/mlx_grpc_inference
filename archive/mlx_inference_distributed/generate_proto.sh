#!/bin/bash
# Generate Python code from Protocol Buffer definitions

echo "Generating Python code from Protocol Buffers..."

# Ensure protos directory exists
mkdir -p protos

# Generate Python code using uv's Python
/opt/homebrew/bin/uv run python -m grpc_tools.protoc \
    -I./protos \
    --python_out=. \
    --grpc_python_out=. \
    ./protos/distributed_inference.proto

# Fix imports in generated files (grpc_tools generates absolute imports)
if [ -f "distributed_inference_pb2_grpc.py" ]; then
    sed -i '' 's/import distributed_inference_pb2/from . import distributed_inference_pb2/' distributed_inference_pb2_grpc.py
fi

echo "Protocol Buffer generation complete!"