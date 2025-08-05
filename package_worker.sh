#!/bin/bash
# Package worker files for deployment

echo "Creating worker deployment package..."

# Create temp directory
mkdir -p /tmp/mlx_worker_deploy/src/communication

# Copy necessary files
cp worker.py /tmp/mlx_worker_deploy/
cp launch_worker.sh /tmp/mlx_worker_deploy/
cp requirements.txt /tmp/mlx_worker_deploy/
cp pyproject.toml /tmp/mlx_worker_deploy/
cp -r src/communication/*.py /tmp/mlx_worker_deploy/src/communication/
cp src/__init__.py /tmp/mlx_worker_deploy/src/
cp src/communication/__init__.py /tmp/mlx_worker_deploy/src/communication/

# Create setup script
cat > /tmp/mlx_worker_deploy/setup_worker.sh << 'EOF'
#!/bin/bash
# Setup worker environment

echo "Setting up MLX worker environment..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create virtual environment and install dependencies
echo "Creating virtual environment..."
uv venv
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install grpcio grpcio-tools mlx mlx-lm fastapi uvicorn pydantic numpy lz4 zstandard

echo "Setup complete!"
echo ""
echo "To start the worker:"
echo "  source .venv/bin/activate"
echo "  ./launch_worker.sh <worker_id> <total_workers>"
echo ""
echo "For example:"
echo "  - On mini2: ./launch_worker.sh 1 3"
echo "  - On m4: ./launch_worker.sh 2 3"
EOF

chmod +x /tmp/mlx_worker_deploy/setup_worker.sh
chmod +x /tmp/mlx_worker_deploy/launch_worker.sh

# Create tarball
cd /tmp
tar -czf mlx_worker_deploy.tar.gz mlx_worker_deploy/

# Move to current directory
mv /tmp/mlx_worker_deploy.tar.gz .

# Cleanup
rm -rf /tmp/mlx_worker_deploy

echo "✓ Worker package created: mlx_worker_deploy.tar.gz"
echo ""
echo "To deploy:"
echo "1. Copy to target machine: scp mlx_worker_deploy.tar.gz user@host:~/"
echo "2. Extract: tar -xzf mlx_worker_deploy.tar.gz"
echo "3. Setup: cd mlx_worker_deploy && ./setup_worker.sh"
echo "4. Run: ./launch_worker.sh <worker_id> 3"