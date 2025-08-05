#!/bin/bash
# Launch coordinator/server on mini1

# Kill any existing processes
pkill -f "python.*server.py"

echo "Starting coordinator server..."
uv run python server.py