#!/bin/bash
# Setup script for master.local to join the distributed MLX cluster

echo "🚀 Setting up master.local for distributed MLX inference..."

# Install uv package manager
echo "📦 Installing uv package manager..."
ssh georgedekker@master.local "curl -LsSf https://astral.sh/uv/install.sh | sh"

# Add uv to PATH
ssh georgedekker@master.local "echo 'export PATH=\"\$HOME/.cargo/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"

# Create project directory
echo "📁 Creating project directory..."
ssh georgedekker@master.local "mkdir -p ~/Movies/mlx_distributed"

# Copy project files
echo "📋 Copying project files..."
scp -r /Users/mini1/Movies/mlx_distributed/*.py georgedekker@master.local:~/Movies/mlx_distributed/
scp -r /Users/mini1/Movies/mlx_distributed/*.json georgedekker@master.local:~/Movies/mlx_distributed/
scp -r /Users/mini1/Movies/mlx_distributed/*.proto georgedekker@master.local:~/Movies/mlx_distributed/
scp -r /Users/mini1/Movies/mlx_distributed/protos georgedekker@master.local:~/Movies/mlx_distributed/
scp -r /Users/mini1/Movies/mlx_distributed/.gitignore georgedekker@master.local:~/Movies/mlx_distributed/

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
ssh georgedekker@master.local "cd ~/Movies/mlx_distributed && ~/.cargo/bin/uv venv && source .venv/bin/activate && ~/.cargo/bin/uv pip install mlx mlx-lm grpcio grpcio-tools protobuf numpy psutil fastapi uvicorn httpx aiohttp"

# Generate protobuf files
echo "🔧 Generating protobuf files..."
ssh georgedekker@master.local "cd ~/Movies/mlx_distributed && source .venv/bin/activate && python -m grpc_tools.protoc -I=protos --python_out=. --grpc_python_out=. protos/distributed_comm.proto"

# Create logs directory
ssh georgedekker@master.local "mkdir -p ~/Movies/mlx_distributed/logs"

echo "✅ Setup complete! Master.local is ready to join the cluster."
echo ""
echo "To start the worker on master.local as rank 2:"
echo "ssh georgedekker@master.local \"cd ~/Movies/mlx_distributed && source .venv/bin/activate && GRPC_DNS_RESOLVER=native LOCAL_RANK=2 DISTRIBUTED_CONFIG=distributed_config.json python worker.py --rank=2\""