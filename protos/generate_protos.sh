#!/bin/bash
# Generate Python code from protocol buffer definitions

cd "$(dirname "$0")"

echo "Generating Python code from proto files..."

cd ..
source .venv/bin/activate
cd protos

python -m grpc_tools.protoc \
    -I. \
    --python_out=../src/communication \
    --grpc_python_out=../src/communication \
    inference.proto

# Fix imports in generated files
cd ../src/communication

# Update imports to use relative imports
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' 's/import inference_pb2/from . import inference_pb2/g' inference_pb2_grpc.py
else
    # Linux
    sed -i 's/import inference_pb2/from . import inference_pb2/g' inference_pb2_grpc.py
fi

echo "Proto generation complete!"