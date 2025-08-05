#!/bin/bash
# Script to run logit comparison test between single-node and distributed inference

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Logit Consistency Test ===${NC}"
echo "This test compares single-node vs distributed inference to ensure numerical consistency"
echo ""

# Default values
MODEL="mlx-community/Qwen3-1.7B-8bit"
MAX_TOKENS=20
TOLERANCE=0.001
WORKER_LOG="logs/worker1_logit_test.log"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --tolerance)
            TOLERANCE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create logs directory
mkdir -p logs

echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Max tokens: $MAX_TOKENS"
echo "  Tolerance: $TOLERANCE"
echo ""

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [[ ! -z "$WORKER_PID" ]]; then
        echo "Stopping worker (PID: $WORKER_PID)"
        kill $WORKER_PID 2>/dev/null || true
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Start worker in background
echo -e "${GREEN}Starting worker 1...${NC}"
MODEL_NAME="$MODEL" RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost LOCAL_TEST=false uv run python server_distributed.py > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!

echo "Worker PID: $WORKER_PID"
echo "Worker log: $WORKER_LOG"

# Wait for worker to be ready
echo -e "${YELLOW}Waiting for worker to initialize...${NC}"
sleep 10

# Check if worker is still running
if ! kill -0 $WORKER_PID 2>/dev/null; then
    echo -e "${RED}Worker failed to start! Check log:${NC}"
    tail -20 "$WORKER_LOG"
    exit 1
fi

# Run the comparison test
echo -e "\n${GREEN}Running logit comparison test...${NC}"
uv run python test_logit_consistency.py \
    --model "$MODEL" \
    --max-tokens $MAX_TOKENS \
    --tolerance $TOLERANCE

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ TEST PASSED!${NC}"
    echo "Single-node and distributed inference are numerically consistent."
else
    echo -e "\n${RED}❌ TEST FAILED!${NC}"
    echo "Check the output above for details."
    echo ""
    echo "Worker log tail:"
    tail -20 "$WORKER_LOG"
fi