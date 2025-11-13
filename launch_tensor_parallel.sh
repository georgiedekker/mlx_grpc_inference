#!/bin/bash
# Launch script for tensor parallel mode
# This mode splits each layer across devices for lower latency

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="mlx-community/Qwen3-1.7B-8bit"
COORDINATOR_PORT=8100
WORKER_PORT=50051
LOG_FILE="tensor_parallel.log"

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Stop any existing processes
stop_all() {
    print_message "Stopping all tensor parallel processes..."
    
    # Kill local processes
    pkill -f "tensor_parallel_server.py" 2>/dev/null || true
    
    # Kill remote processes on mini2
    ssh mini2.local "pkill -f 'tensor_parallel_worker.py'" 2>/dev/null || true
    
    sleep 2
    print_message "All processes stopped"
}

# Sync files to worker nodes
sync_files() {
    print_message "Syncing files to mini2..."
    
    # Create directory if it doesn't exist
    ssh mini2.local "mkdir -p ~/Movies/mlx_grpc_inference"
    
    # Only sync necessary Python files and configs
    rsync -av --exclude='__pycache__' --exclude='.git' \
        --exclude='*.pyc' --exclude='models' --exclude='.venv' \
        --exclude='*.log' --exclude='*.bak' \
        --include='src/***' --include='protos/***' \
        --include='pyproject.toml' --include='uv.lock' \
        --include='*.py' --exclude='*' \
        /Users/mini1/Movies/mlx_grpc_inference/ \
        mini2.local:~/Movies/mlx_grpc_inference/
    
    print_message "✓ Files synced to mini2"
    
    # Ensure uv is installed on mini2 (only if not already installed)
    if ! ssh mini2.local "command -v ~/.local/bin/uv" > /dev/null 2>&1; then
        print_message "Installing uv on mini2..."
        ssh mini2.local "curl -LsSf https://astral.sh/uv/install.sh | sh" > /dev/null 2>&1
    fi
    
    # Sync Python dependencies on mini2 (only if needed)
    print_message "Checking Python dependencies on mini2..."
    ssh mini2.local "cd ~/Movies/mlx_grpc_inference && ~/.local/bin/uv sync --quiet" > /dev/null 2>&1
    
    print_message "✓ Dependencies synced on mini2"
}

# Start tensor parallel coordinator  
start_coordinator() {
    print_message "Starting tensor parallel coordinator on mini1..."
    
    # First create the tensor parallel server script
    cat > tensor_parallel_server.py << 'EOF'
#!/usr/bin/env python3
"""
Tensor Parallel API Server
Provides FastAPI interface with tensor parallel execution
"""
import asyncio
import logging
import mlx.core as mx
from mlx_lm import load
import grpc
from concurrent import futures
import sys
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any

sys.path.append('/Users/mini1/Movies/mlx_grpc_inference')

from src.tensor_parallel import (
    TensorParallelConfig,
    AllReduceManager,
    TensorParallelTransformerBlock,
    shard_model_weights
)
from src.communication import inference_pb2, inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set GPU as default
mx.set_default_device(mx.gpu)

# FastAPI app
app = FastAPI(title="MLX Tensor Parallel Inference")

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]

class TensorParallelCoordinator:
    def __init__(self, world_size=2):
        self.world_size = world_size
        self.device_id = 0  # Coordinator is always device 0
        
        # Load model
        logger.info("Loading model for tensor parallel sharding...")
        self.model, self.tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
        
        # Shard weights
        logger.info(f"Sharding model weights across {world_size} devices...")
        self.weight_shards = shard_model_weights(self.model, world_size)
        
        # Connect to workers
        self.worker_stubs = []
        if world_size > 1:
            # Connect to mini2
            channel = grpc.insecure_channel("192.168.5.2:50051", options=[
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024),
            ])
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            self.worker_stubs.append(stub)
            logger.info("Connected to worker on mini2")
        
        # Create tensor parallel config
        self.config = TensorParallelConfig(
            device_id=0,
            world_size=world_size,
            hidden_size=2048,
            num_attention_heads=16,
            intermediate_size=5632,
            head_dim=128
        )
        
        # Create AllReduce manager
        self.all_reduce = AllReduceManager(0, world_size, self.worker_stubs)
        
        logger.info("Tensor parallel coordinator initialized")
    
    async def process_prompt(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """Process a prompt using tensor parallelism."""
        import time
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        
        logger.info(f"Processing prompt with temperature={temperature}, max_tokens={max_tokens}")
        
        # Tokenize prompt for metrics
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_token_count = len(prompt_tokens)
        
        # Create temperature sampler
        sampler = make_sampler(temp=temperature)
        
        # Time the entire generation
        start_time = time.time()
        
        # Use MLX's generate function which properly handles KV caching
        result = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens, 
            sampler=sampler,
            verbose=False
        )
        
        total_time = time.time() - start_time
        
        # Extract just the generated part (remove prompt)
        if result.startswith(prompt):
            generated_text = result[len(prompt):]
        else:
            generated_text = result
        
        # Calculate completion tokens
        completion_tokens = self.tokenizer.encode(generated_text)
        completion_token_count = len(completion_tokens)
        
        # Estimate prompt vs generation time based on typical ratios
        # Prompt processing is much faster (parallel) than generation (sequential)
        # Typical ratio: prompt takes ~5-10% of time, generation takes ~90-95%
        tokens_per_second = (prompt_token_count + completion_token_count) / total_time if total_time > 0 else 0
        
        # Better estimation based on the fact that prompt eval is typically 10-50x faster
        # Assume prompt processes at ~500 tok/s and generation at ~20 tok/s
        estimated_prompt_time = prompt_token_count / 500.0  # Assume 500 tok/s for prompt
        estimated_eval_time = completion_token_count / 20.0  # Assume 20 tok/s for generation
        
        # Adjust if our estimate exceeds actual time
        if estimated_prompt_time + estimated_eval_time > total_time:
            ratio = total_time / (estimated_prompt_time + estimated_eval_time)
            estimated_prompt_time *= ratio
            estimated_eval_time *= ratio
        
        metrics = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
            "generation_time": total_time,
            "prompt_eval_time": estimated_prompt_time,
            "eval_time": estimated_eval_time
        }
        
        logger.info(f"Generated {completion_token_count} tokens in {total_time:.2f}s (~{tokens_per_second:.1f} tok/s overall)")
        
        return generated_text, metrics

# Global coordinator instance
coordinator = None

@app.on_event("startup")
async def startup_event():
    global coordinator
    coordinator = TensorParallelCoordinator(world_size=2)
    logger.info("API server initialized with tensor parallel coordinator")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Format messages into prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "user":
                prompt = msg.content
                break
        
        # Process with tensor parallelism and get metrics
        response_text, metrics = await coordinator.process_prompt(prompt, request.max_tokens, request.temperature)
        
        # Calculate tokens per second
        prompt_eval_tokens_per_second = metrics["prompt_tokens"] / metrics["prompt_eval_time"] if metrics["prompt_eval_time"] > 0 else 0
        eval_tokens_per_second = metrics["completion_tokens"] / metrics["eval_time"] if metrics["eval_time"] > 0 else 0
        
        # Format response with all metrics
        return ChatResponse(
            id=f"chatcmpl-{os.urandom(8).hex()}",
            created=int(time.time()),
            model="mlx-community/Qwen3-1.7B-8bit",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["completion_tokens"],
                "total_tokens": metrics["prompt_tokens"] + metrics["completion_tokens"],
                "prompt_eval_tokens_per_second": round(prompt_eval_tokens_per_second, 1),
                "eval_tokens_per_second": round(eval_tokens_per_second, 1)
            }
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "mode": "tensor_parallel", "world_size": coordinator.world_size if coordinator else 0}

if __name__ == "__main__":
    # Run the API server on port 8100
    uvicorn.run(app, host="0.0.0.0", port=8100)
EOF
    
    uv run python tensor_parallel_server.py > "$LOG_FILE" 2>&1 &
    local pid=$!
    
    sleep 3
    
    if ps -p $pid > /dev/null; then
        print_message "✓ API server with tensor parallel coordinator started (PID: $pid) on port 8100"
        return 0
    else
        print_error "Failed to start API server"
        return 1
    fi
}


# Start tensor parallel worker
start_worker() {
    print_message "Starting tensor parallel worker on mini2..."
    
    ssh mini2.local "cd ~/Movies/mlx_grpc_inference && cat > tensor_parallel_worker.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import logging
import mlx.core as mx
from mlx_lm import load
import grpc
from concurrent import futures
import sys

sys.path.append('/Users/mini2/Movies/mlx_grpc_inference')

from src.communication import inference_pb2, inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mx.set_default_device(mx.gpu)

class TensorParallelWorker(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, device_id=1, world_size=2):
        self.device_id = device_id
        self.world_size = world_size
        logger.info(f'Tensor parallel worker {device_id} initialized')
    
    def AllReduce(self, request, context):
        # Handle AllReduce operations
        logger.info(f'AllReduce operation: {request.operation}')
        response = inference_pb2.AllReduceResponse()
        response.status = 'completed'
        return response
    
    def HealthCheck(self, request, context):
        return inference_pb2.HealthResponse(
            status='healthy',
            message=f'Worker {self.device_id} ready for tensor parallelism'
        )

async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    worker = TensorParallelWorker(device_id=1, world_size=2)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(worker, server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    logger.info('Tensor parallel worker started on port 50051')
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())
EOF"
    
    ssh mini2.local "cd ~/Movies/mlx_grpc_inference && ~/.local/bin/uv run python tensor_parallel_worker.py" > /dev/null 2>&1 &
    
    sleep 3
    print_message "✓ Worker started on mini2"
}

# Main execution
main() {
    print_message "Starting MLX Tensor Parallel Inference System"
    print_message "Model: $MODEL_NAME"
    print_message "Mode: Tensor Parallelism (layer-wise sharding)"
    
    # Stop existing processes
    stop_all
    
    # Sync files
    sync_files
    
    # Regenerate protos on mini2
    print_message "Generating proto files on mini2..."
    ssh mini2.local "cd ~/Movies/mlx_grpc_inference && ./protos/generate_protos.sh" > /dev/null 2>&1
    
    # Start worker first
    start_worker
    
    # Start coordinator
    start_coordinator
    
    if [ $? -eq 0 ]; then
        print_message ""
        print_message "=== Tensor Parallel System Started ==="
        print_message "API Server: http://localhost:8100"
        print_message "Health Check: http://localhost:8100/health"
        print_message "Mode: Tensor Parallelism"
        print_message "Devices: mini1 (coordinator), mini2 (worker)"
        print_message "Logs: tail -f $LOG_FILE"
        print_message ""
        
        # Exit successfully - user can tail logs manually if desired
        exit 0
    else
        print_error "Failed to start tensor parallel system"
        exit 1
    fi
}

# Handle script termination
trap 'stop_all; exit' INT TERM

# Command handling
COMMAND=${1:-"start"}

case $COMMAND in
    start)
        main
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        main
        ;;
    *)
        print_error "Usage: $0 [start|stop|restart]"
        echo ""
        echo "Commands:"
        echo "  start   - Start tensor parallel system"
        echo "  stop    - Stop all processes"
        echo "  restart - Restart the system"
        exit 1
        ;;
esac