#!/bin/bash
# Launch script for 2-device MLX distributed cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="${CONFIG_FILE:-distributed_config.json}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${PROJECT_DIR}/logs"

echo -e "${GREEN}🚀 Starting 2-Device MLX Distributed Cluster${NC}"
echo "  Project Directory: $PROJECT_DIR"
echo "  Configuration: $CONFIG_FILE"
echo "  Logs Directory: $LOGS_DIR"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Create logs directory
mkdir -p "$LOGS_DIR"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up cluster processes...${NC}"
    
    # Kill API server
    if [ -f "$LOGS_DIR/api_server.pid" ]; then
        API_PID=$(cat "$LOGS_DIR/api_server.pid")
        echo "Stopping API server (PID: $API_PID)"
        kill -TERM "$API_PID" 2>/dev/null || true
        rm -f "$LOGS_DIR/api_server.pid"
    fi
    
    # Kill mini2 worker
    if [ -f "$LOGS_DIR/mini2_worker.pid" ]; then
        WORKER_PID=$(cat "$LOGS_DIR/mini2_worker.pid")
        echo "Stopping mini2 worker (PID: $WORKER_PID)"
        kill -TERM "$WORKER_PID" 2>/dev/null || true
        rm -f "$LOGS_DIR/mini2_worker.pid"
    fi
    
    # Kill any remaining processes
    pkill -f "run_distributed_openai.py" 2>/dev/null || true
    pkill -f "worker.py" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# Step 1: Test network connectivity
echo -e "${BLUE}📡 Testing network connectivity...${NC}"
if ! uv run python test_network_setup.py; then
    echo -e "${RED}❌ Network connectivity test failed!${NC}"
    echo "Please ensure:"
    echo "  1. mini2.local is reachable: ping mini2.local"
    echo "  2. SSH keys are set up (if using SSH deployment)"
    echo "  3. Firewall allows gRPC ports (50100-50102, 8100)"
    exit 1
fi
echo -e "${GREEN}✅ Network connectivity verified${NC}"
echo ""

# Step 2: Start worker on mini2
echo -e "${BLUE}🔧 Starting worker on mini2...${NC}"

# Check if we should use SSH or local execution
USE_SSH=${USE_SSH:-true}

if [ "$USE_SSH" = "true" ]; then
    # Use SSH to start worker on mini2
    echo "Using SSH to start worker on mini2.local..."
    
    # Copy project files to mini2 if needed
    echo "Syncing project files to mini2..."
    rsync -avz --exclude 'logs' --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
        "$PROJECT_DIR/" "mini2.local:~/Movies/mlx_distributed/" || {
        echo -e "${RED}❌ Failed to sync files to mini2${NC}"
        exit 1
    }
    
    # Start worker on mini2 via SSH
    ssh mini2.local "cd ~/Movies/mlx_distributed && nohup uv run python worker.py --rank 1 > logs/mini2_worker.log 2>&1 &" || {
        echo -e "${RED}❌ Failed to start worker on mini2 via SSH${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Worker started on mini2 via SSH${NC}"
else
    # Local execution (for testing on single machine)
    echo "Starting worker locally (rank 1)..."
    LOCAL_RANK=1 uv run python worker.py --rank 1 > "$LOGS_DIR/mini2_worker.log" 2>&1 &
    WORKER_PID=$!
    echo "$WORKER_PID" > "$LOGS_DIR/mini2_worker.pid"
    echo -e "${GREEN}✅ Worker started locally (PID: $WORKER_PID)${NC}"
fi

# Give worker time to initialize
echo "Waiting for worker to initialize..."
sleep 5

# Step 3: Start coordinator/API server on mini1
echo -e "${BLUE}🎯 Starting coordinator and API server on mini1...${NC}"

# Export environment variables
export LOCAL_RANK=0
export DISTRIBUTED_CONFIG="$CONFIG_FILE"

# Start the distributed API server
uv run python run_distributed_openai.py > "$LOGS_DIR/api_server.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "$LOGS_DIR/api_server.pid"

echo -e "${GREEN}✅ API server started (PID: $API_PID)${NC}"

# Give API server time to initialize
echo "Waiting for API server to initialize..."
sleep 10

# Step 4: Verify cluster is running
echo -e "${BLUE}🔍 Verifying cluster status...${NC}"

# Test API health
for i in {1..10}; do
    if curl -s "http://localhost:8100/health" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ API server is responding${NC}"
        break
    elif [ $i -eq 10 ]; then
        echo -e "${RED}❌ API server is not responding after 10 attempts${NC}"
        echo "Check logs: tail -f $LOGS_DIR/api_server.log"
        exit 1
    else
        echo "Waiting for API server... (attempt $i/10)"
        sleep 2
    fi
done

# Test cluster status
echo ""
echo -e "${BLUE}📊 Cluster Status:${NC}"
if curl -s "http://localhost:8100/distributed/status" | python -m json.tool 2>/dev/null; then
    echo ""
else
    echo -e "${YELLOW}⚠️  Could not get cluster status via API${NC}"
fi

# Show GPU info
echo -e "${BLUE}🖥️  GPU Information:${NC}"
if curl -s "http://localhost:8100/distributed/gpu-info" | python -m json.tool 2>/dev/null; then
    echo ""
else
    echo -e "${YELLOW}⚠️  Could not get GPU info via API${NC}"
fi

# Step 5: Success message and instructions
echo -e "${GREEN}🎉 2-Device MLX Distributed Cluster is running!${NC}"
echo ""
echo -e "${BLUE}📋 Cluster Information:${NC}"
echo "  • Coordinator (mini1): http://localhost:8100"
echo "  • Worker (mini2): Rank 1"
echo "  • API Endpoints:"
echo "    - Health: http://localhost:8100/health"
echo "    - Models: http://localhost:8100/v1/models"
echo "    - Chat: http://localhost:8100/v1/chat/completions"
echo "    - Status: http://localhost:8100/distributed/status"
echo "    - GPU Info: http://localhost:8100/distributed/gpu-info"
echo ""
echo -e "${BLUE}📝 Log Files:${NC}"
echo "  • API Server: $LOGS_DIR/api_server.log"
echo "  • mini2 Worker: $LOGS_DIR/mini2_worker.log"
echo ""
echo -e "${BLUE}🧪 Test Commands:${NC}"
echo '  # Test basic health'
echo '  curl http://localhost:8100/health'
echo ''
echo '  # Test chat completion'
echo '  curl -X POST http://localhost:8100/v1/chat/completions \'
echo '       -H "Content-Type: application/json" \'
echo '       -d "{\"model\": \"mlx-community/Qwen3-1.7B-8bit\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 50}"'
echo ""
echo -e "${YELLOW}⚠️  Press Ctrl+C to stop the cluster${NC}"

# Wait for interrupt
wait $API_PID