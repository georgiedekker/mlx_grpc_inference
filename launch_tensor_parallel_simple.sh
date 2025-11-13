#!/bin/bash
# Simplified launch script for tensor parallel mode
# Uses your working tensor parallel setup without breaking it

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Just use the existing working script
print_message() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if tensor parallel is already running
if pgrep -f "tensor_parallel_server.py" > /dev/null; then
    print_message "Tensor parallel system is already running"
    print_info "API Server: http://localhost:8100"
    print_info "Health Check: http://localhost:8100/health"
    exit 0
fi

# Launch the working tensor parallel system
print_message "Launching tensor parallel system..."
./launch_tensor_parallel.sh start

# Add monitoring info
print_message ""
print_info "System launched successfully!"
print_info "The monitoring dashboard is integrated into the API server"
print_info ""
print_info "Monitor performance by checking the logs:"
print_info "  tail -f tensor_parallel.log"
print_info ""
print_info "Test the system:"
print_info '  curl -X POST "http://localhost:8100/v1/chat/completions" \'
print_info '    -H "Content-Type: application/json" \'
print_info '    -d '\''{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20}'\'''