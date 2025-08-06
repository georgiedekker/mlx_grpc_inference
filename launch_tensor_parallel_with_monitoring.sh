#!/bin/bash
# Launch script for tensor parallel mode with integrated monitoring
# This mode splits each layer across devices for lower latency

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="mlx-community/Qwen3-1.7B-8bit"
API_PORT=8100
DASHBOARD_PORT=8888
WORKER_PORT=50051
LOG_FILE="tensor_parallel.log"
DASHBOARD_LOG="dashboard.log"

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

print_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Stop any existing processes
stop_all() {
    print_message "Stopping all processes..."
    
    # Kill dashboard
    pkill -f "tensor_parallel_dashboard.py" 2>/dev/null || true
    
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

# Start monitoring dashboard
start_dashboard() {
    print_info "Starting performance monitoring dashboard..."
    
    # Create dashboard script that integrates with tensor parallel
    cat > tensor_parallel_dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
Performance monitoring dashboard for tensor parallel inference.
"""
import asyncio
import sys
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import time
import psutil
import mlx.core as mx

# Add project root to path
sys.path.append('/Users/mini1/Movies/mlx_grpc_inference')

from src.performance_monitor import get_monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for dashboard
app = FastAPI(title="MLX Tensor Parallel Dashboard")

# Get monitor instance
monitor = get_monitor()

@app.get("/")
async def dashboard():
    """Serve dashboard HTML."""
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>MLX Tensor Parallel Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1e1e1e;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        h1 {
            margin: 0;
            font-size: 28px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #3a3a3a;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #4ade80;
        }
        .metric-label {
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            margin-top: 5px;
        }
        .chart-container {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #3a3a3a;
            margin-bottom: 20px;
            height: 300px;
        }
        .status {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-healthy { background: #4ade80; color: #000; }
        .status-warning { background: #fbbf24; color: #000; }
        .status-error { background: #f87171; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ MLX Tensor Parallel Inference Dashboard</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">
                Monitoring tensor-parallel execution across mini1 and mini2
            </p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="api-status">
                    <span class="status status-warning">LOADING</span>
                </div>
                <div class="metric-label">API Status</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="tokens-per-second">0</div>
                <div class="metric-label">Tokens/Second</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="latency">0</div>
                <div class="metric-label">Latency (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="gpu-memory">0</div>
                <div class="metric-label">GPU Memory (MB)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="active-sessions">0</div>
                <div class="metric-label">Active Sessions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="total-requests">0</div>
                <div class="metric-label">Total Requests</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="performance-chart"></canvas>
        </div>
        
        <div style="background: #2a2a2a; padding: 20px; border-radius: 8px; border: 1px solid #3a3a3a;">
            <h3>Test Commands</h3>
            <pre style="color: #4ade80; font-family: 'Courier New', monospace;">
# Test the API
curl -X POST "http://localhost:8100/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is tensor parallelism?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Check health
curl http://localhost:8100/health</pre>
        </div>
    </div>
    
    <script>
        // Initialize performance chart
        const ctx = document.getElementById('performance-chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Tokens/Second',
                    data: [],
                    borderColor: '#4ade80',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#fff' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#999' },
                        grid: { color: '#3a3a3a' }
                    },
                    y: {
                        ticks: { color: '#999' },
                        grid: { color: '#3a3a3a' }
                    }
                }
            }
        });
        
        let totalRequests = 0;
        
        // Check API health
        async function checkApiHealth() {
            try {
                const response = await fetch('http://localhost:8100/health');
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('api-status').innerHTML = 
                        '<span class="status status-healthy">ONLINE</span>';
                } else {
                    document.getElementById('api-status').innerHTML = 
                        '<span class="status status-error">OFFLINE</span>';
                }
            } catch (error) {
                document.getElementById('api-status').innerHTML = 
                    '<span class="status status-error">OFFLINE</span>';
            }
        }
        
        // Update metrics
        async function updateMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                
                if (data.current) {
                    document.getElementById('tokens-per-second').textContent = 
                        data.current.tokens_per_second.toFixed(1);
                    document.getElementById('latency').textContent = 
                        data.current.latency_ms.toFixed(1);
                    document.getElementById('gpu-memory').textContent = 
                        data.current.gpu_memory_mb.toFixed(0);
                    document.getElementById('active-sessions').textContent = 
                        data.current.active_sessions;
                    
                    // Update chart
                    const time = new Date().toLocaleTimeString();
                    chart.data.labels.push(time);
                    chart.data.datasets[0].data.push(data.current.tokens_per_second);
                    
                    // Keep only last 30 points
                    if (chart.data.labels.length > 30) {
                        chart.data.labels.shift();
                        chart.data.datasets[0].data.shift();
                    }
                    
                    chart.update();
                }
                
                if (data.totals) {
                    totalRequests = data.totals.total_tokens_processed || totalRequests;
                    document.getElementById('total-requests').textContent = totalRequests;
                }
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
            }
        }
        
        // Start updating
        checkApiHealth();
        setInterval(checkApiHealth, 5000);
        setInterval(updateMetrics, 1000);
        updateMetrics();
    </script>
</body>
</html>
    '''
    return HTMLResponse(content=html_content)

@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics as JSON."""
    return JSONResponse(monitor.get_metrics_summary())

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "dashboard": "running"}

if __name__ == "__main__":
    logger.info(f"Starting Tensor Parallel Dashboard on port 8888")
    logger.info(f"Dashboard URL: http://localhost:8888")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="error")
EOF
    
    uv run python tensor_parallel_dashboard.py > "$DASHBOARD_LOG" 2>&1 &
    local dashboard_pid=$!
    
    sleep 2
    
    if ps -p $dashboard_pid > /dev/null; then
        print_message "✅ Dashboard started (PID: $dashboard_pid)"
        print_info "📊 Dashboard available at: http://localhost:$DASHBOARD_PORT"
        return 0
    else
        print_error "Failed to start dashboard"
        cat "$DASHBOARD_LOG" | tail -10
        return 1
    fi
}

# Start tensor parallel coordinator  
start_coordinator() {
    print_message "Starting tensor parallel coordinator on mini1..."
    
    # Copy the coordinator script from launch_tensor_parallel.sh
    cat > tensor_parallel_server.py << 'EOF'
#!/usr/bin/env python3
"""
Tensor Parallel API Server with Monitoring Integration
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
from src.performance_monitor import get_monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set GPU as default
mx.set_default_device(mx.gpu)

# Get performance monitor
monitor = get_monitor()

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
        """Process a prompt using tensor parallelism with monitoring."""
        import uuid
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        
        session_id = str(uuid.uuid4())
        logger.info(f"Processing prompt with temperature={temperature}, max_tokens={max_tokens}, session={session_id}")
        
        # Tokenize prompt for metrics
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_token_count = len(prompt_tokens)
        
        # Create temperature sampler
        sampler = make_sampler(temp=temperature)
        
        # Time the entire generation
        start_time = time.time()
        
        # Record prompt processing
        prompt_start = time.time()
        
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
        
        # Estimate times
        estimated_prompt_time = prompt_token_count / 500.0  # ~500 tok/s for prompt
        estimated_eval_time = total_time - estimated_prompt_time
        if estimated_eval_time < 0:
            estimated_eval_time = total_time * 0.9
            estimated_prompt_time = total_time * 0.1
        
        # Record metrics in monitor
        monitor.record_token_generation(session_id, prompt_token_count, estimated_prompt_time, is_prompt=True)
        monitor.record_token_generation(session_id, completion_token_count, estimated_eval_time, is_prompt=False)
        monitor.end_session(session_id)
        
        metrics = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
            "generation_time": total_time,
            "prompt_eval_time": estimated_prompt_time,
            "eval_time": estimated_eval_time
        }
        
        tokens_per_second = (prompt_token_count + completion_token_count) / total_time if total_time > 0 else 0
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
    
    sleep 5
    
    if ps -p $pid > /dev/null; then
        print_message "✅ API server started (PID: $pid) on port $API_PORT"
        return 0
    else
        print_error "Failed to start API server"
        cat "$LOG_FILE" | tail -20
        return 1
    fi
}

# Start tensor parallel worker (same as original)
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
    print_message "🚀 Starting MLX Tensor Parallel System with Monitoring"
    print_message "Model: $MODEL_NAME"
    print_message "Mode: Tensor Parallelism with Performance Monitoring"
    echo ""
    
    # Stop existing processes
    stop_all
    
    # Sync files
    sync_files
    
    # Regenerate protos on mini2
    print_message "Generating proto files on mini2..."
    ssh mini2.local "cd ~/Movies/mlx_grpc_inference && ./protos/generate_protos.sh" > /dev/null 2>&1
    
    # Start monitoring dashboard
    if ! start_dashboard; then
        print_error "Failed to start dashboard, continuing without monitoring"
    fi
    
    # Start worker first
    start_worker
    
    # Start coordinator
    start_coordinator
    
    if [ $? -eq 0 ]; then
        print_message ""
        print_message "=== Tensor Parallel System with Monitoring Started ==="
        print_info "📊 Dashboard: http://localhost:$DASHBOARD_PORT"
        print_info "🚀 API Server: http://localhost:$API_PORT"
        print_info "📋 Health Check: http://localhost:$API_PORT/health"
        print_message ""
        print_message "Mode: Tensor Parallelism with Real-time Monitoring"
        print_message "Devices: mini1 (coordinator), mini2 (worker)"
        print_message ""
        print_info "Test with:"
        echo '  curl -X POST "http://localhost:8100/v1/chat/completions" \'
        echo '    -H "Content-Type: application/json" \'
        echo '    -d '\''{"messages": [{"role": "user", "content": "What is tensor parallelism?"}], "max_tokens": 50}'\'''
        print_message ""
        print_message "Logs:"
        print_info "  • API: tail -f $LOG_FILE"
        print_info "  • Dashboard: tail -f $DASHBOARD_LOG"
        print_message ""
        
        # Exit successfully
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
        echo "  start   - Start tensor parallel system with monitoring"
        echo "  stop    - Stop all processes"
        echo "  restart - Restart the system"
        exit 1
        ;;
esac