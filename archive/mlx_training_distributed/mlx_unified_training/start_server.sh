#!/bin/bash
# Quick start script for MLX Unified Training Platform

echo "🚀 Starting MLX Unified Training Platform on port 8600..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
    echo "📥 Installing dependencies..."
    uv pip install fastapi uvicorn pydantic
fi

# Set API key
export MLX_UNIFIED_API_KEY="mlx-unified-key"

# Start the server
echo "🌐 Server starting on http://localhost:8600"
echo "📡 API documentation: http://localhost:8600/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uv run python src/api/server.py