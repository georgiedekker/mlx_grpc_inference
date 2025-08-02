#!/bin/bash
# Setup script for MLX Unified Training Platform

echo "🚀 Setting up MLX Unified Training Platform..."
echo "============================================"

# Create virtual environment
echo "📦 Creating virtual environment..."
uv venv

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -e .

# Set up environment variables
echo "🔧 Setting up environment..."
export MLX_UNIFIED_API_KEY="mlx-unified-key"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the server:"
echo "  uv run python src/api/server.py"
echo ""
echo "The server will run on http://localhost:8600"
echo ""
echo "To run examples:"
echo "  uv run python examples/unified_training_example.py"