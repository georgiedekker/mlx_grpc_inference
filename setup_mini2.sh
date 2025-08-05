#!/bin/bash
# Setup script to prepare mini2 for distributed inference

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up mini2 for distributed inference${NC}"

# Install uv on mini2
echo -e "${GREEN}Installing uv on mini2...${NC}"
ssh mini2@mini2.local 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# Add uv to PATH
echo -e "${GREEN}Adding uv to PATH...${NC}"
ssh mini2@mini2.local 'echo "export PATH=\"\$HOME/.cargo/bin:\$PATH\"" >> ~/.zshrc'

# Create project directory
echo -e "${GREEN}Creating project directory...${NC}"
ssh mini2@mini2.local 'mkdir -p /Users/mini2/Movies/mlx_grpc_inference'

# Sync project files
echo -e "${GREEN}Syncing project files...${NC}"
rsync -az --exclude='logs' --exclude='__pycache__' --exclude='.venv' \
    /Users/mini1/Movies/mlx_grpc_inference/ \
    mini2@mini2.local:/Users/mini2/Movies/mlx_grpc_inference/

# Initialize uv project on mini2
echo -e "${GREEN}Initializing uv project on mini2...${NC}"
ssh mini2@mini2.local 'cd /Users/mini2/Movies/mlx_grpc_inference && source ~/.zshrc && uv sync'

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}mini2 is now ready for distributed inference${NC}"