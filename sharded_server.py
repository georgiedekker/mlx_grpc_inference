#!/usr/bin/env python3
"""
MLX Distributed Sharded Inference Server
Uses layer-based sharding to distribute model across GPUs
Based on exo's approach but simplified for direct use
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional
import uuid
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.sample_utils import make_sampler

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from shard import Shard
from sharded_model_loader import load_sharded_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU as default
mx.set_default_device(mx.gpu)

# Global state
model = None
tokenizer = None
distributed_group = None
rank = 0
world_size = 1
shard = None
model_name = "mlx-community/Qwen3-1.7B-8bit"
model_path = None

# Model configuration (Qwen3-1.7B has 28 layers)
TOTAL_LAYERS = 28


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


def get_model_path(model_name: str) -> Path:
    """Get the local path for a model"""
    # MLX models are stored in ~/.cache/huggingface/hub/
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Convert model name to cache format
    model_id = f"models--{model_name.replace('/', '--')}"
    model_dir = cache_dir / model_id
    
    if not model_dir.exists():
        logger.error(f"Model not found at {model_dir}")
        logger.info(f"Please download the model first using mlx_lm")
        return None
    
    # Find the snapshot directory
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        # Get the latest snapshot
        snapshot_dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
        if snapshot_dirs:
            return snapshot_dirs[-1]
    
    return None


def load_model_sharded():
    """Load model with proper layer sharding across devices"""
    global model, tokenizer, distributed_group, rank, world_size, shard, model_path
    
    # Initialize distributed
    distributed_group = mx.distributed.init()
    
    if not distributed_group:
        logger.error("No distributed group found. Please run with mpirun")
        sys.exit(1)
    
    rank = distributed_group.rank()
    world_size = distributed_group.size()
    hostname = os.uname().nodename
    
    logger.info(f"🚀 Rank {rank}/{world_size} on {hostname}")
    
    if world_size != 2:
        logger.error(f"This script requires exactly 2 GPUs, got {world_size}")
        sys.exit(1)
    
    # Get model path
    model_path = get_model_path(model_name)
    if not model_path:
        logger.error(f"Could not find model {model_name}")
        sys.exit(1)
    
    logger.info(f"Model path: {model_path}")
    
    # Define shards for each rank
    # Split layers evenly between devices
    layers_per_device = TOTAL_LAYERS // world_size
    
    if rank == 0:
        # First device: layers 0-13 + embeddings
        shard = Shard(
            model_id=model_name,
            start_layer=0,
            end_layer=layers_per_device - 1,
            n_layers=TOTAL_LAYERS
        )
        logger.info(f"Rank 0: Loading layers {shard.start_layer}-{shard.end_layer} (first shard)")
    else:
        # Second device: layers 14-27 + LM head
        shard = Shard(
            model_id=model_name,
            start_layer=layers_per_device,
            end_layer=TOTAL_LAYERS - 1,
            n_layers=TOTAL_LAYERS
        )
        logger.info(f"Rank 1: Loading layers {shard.start_layer}-{shard.end_layer} (last shard)")
    
    # Load sharded model
    logger.info(f"Rank {rank}: Loading sharded model...")
    model = load_sharded_model(str(model_path), shard, lazy=False)
    
    # Load tokenizer (both ranks need it)
    tokenizer = load_tokenizer(str(model_path))
    
    # Log memory usage
    gpu_memory = mx.get_active_memory() / (1024**3)
    logger.info(f"Rank {rank}: GPU memory used: {gpu_memory:.2f} GB")
    
    # Synchronize
    logger.info(f"Rank {rank}: Synchronizing...")
    sync = mx.distributed.all_sum(mx.array([1.0]))
    mx.eval(sync)
    logger.info(f"Rank {rank}: Ready")
    
    if rank == 0:
        logger.info("="*60)
        logger.info("✅ SHARDED INFERENCE READY")
        logger.info(f"✅ Model: {model_name}")
        logger.info(f"✅ Devices: {world_size} GPUs")
        logger.info(f"✅ Rank 0: Layers 0-{layers_per_device-1} + embeddings")
        logger.info(f"✅ Rank 1: Layers {layers_per_device}-{TOTAL_LAYERS-1} + LM head")
        logger.info("="*60)


def distributed_forward(inputs: mx.array, cache=None) -> mx.array:
    """Forward pass through distributed model"""
    global model, rank, world_size, distributed_group
    
    if world_size == 1:
        # Single device, just run normally
        return model(inputs, cache)
    
    # Rank 0: Process first half of layers
    if rank == 0:
        # Process through our layers
        output = model(inputs, cache)
        
        # Send output to rank 1
        # MLX distributed doesn't have send/recv, so we use all_sum with zeros
        # This is a workaround - in production you'd want proper send/recv
        output_flat = output.reshape(-1)
        
        # Pad if needed for all_sum
        gathered = mx.distributed.all_sum(output_flat, group=distributed_group)
        
        # Rank 0 returns None (not the final node)
        return None
    
    # Rank 1: Receive from rank 0 and process second half
    else:
        # Receive input from rank 0 via all_sum trick
        # This assumes rank 0 sent the data
        input_shape = inputs.shape  # We need to know expected shape
        input_size = inputs.size
        
        received = mx.distributed.all_sum(mx.zeros(input_size), group=distributed_group)
        received = received.reshape(input_shape)
        
        # Process through our layers
        output = model(received, cache)
        
        # Rank 1 has the final output
        return output


def generate_distributed(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
    """Generate text using distributed model"""
    global model, tokenizer, rank, world_size
    
    # Only rank 0 handles tokenization and generation orchestration
    if rank == 0:
        # Tokenize prompt
        tokens = tokenizer.encode(prompt)
        prompt_tokens = mx.array(tokens).reshape(1, -1)
        
        logger.info(f"Generating {max_tokens} tokens from {len(tokens)} prompt tokens")
        
        start_time = time.time()
        generated_tokens = []
        
        # Create sampler
        sampler = make_sampler(temperature)
        
        # Generate tokens
        for _ in range(max_tokens):
            # Forward pass through distributed model
            logits = distributed_forward(prompt_tokens)
            
            if logits is not None:
                # Should not happen on rank 0
                logger.error("Rank 0 got logits but shouldn't")
            
            # Wait for rank 1 to send back the token
            # This is a simplified approach - in production you'd want proper communication
            token = mx.distributed.all_sum(mx.array([0]), group=distributed_group)
            
            generated_tokens.append(int(token.item()))
            
            # Check for EOS
            if int(token.item()) == tokenizer.eos_token_id:
                break
            
            # Prepare next input
            prompt_tokens = token.reshape(1, -1)
        
        # Calculate metrics
        total_time = time.time() - start_time
        tokens_per_second = len(generated_tokens) / total_time if total_time > 0 else 0
        
        # Decode generated text
        full_tokens = tokens + generated_tokens
        generated_text = tokenizer.decode(generated_tokens)
        
        logger.info(f"✅ Generated {len(generated_tokens)} tokens in {total_time:.2f}s")
        logger.info(f"✅ Speed: {tokens_per_second:.1f} tokens/sec")
        
        return {
            "text": generated_text,
            "tokens_generated": len(generated_tokens),
            "tokens_per_second": tokens_per_second,
            "time_taken": total_time
        }
    
    # Rank 1: Process forward passes and sampling
    else:
        for _ in range(max_tokens):
            # Receive dummy input to know we should process
            _ = mx.distributed.all_sum(mx.array([0]), group=distributed_group)
            
            # Forward pass
            logits = distributed_forward(mx.array([[0]]))  # Dummy input, will be replaced
            
            if logits is None:
                logger.error("Rank 1 got None but should have logits")
                break
            
            # Sample next token
            sampler = make_sampler(temperature)
            logits = logits[:, -1, :]
            token = sampler(logits)
            
            # Send token back to rank 0
            mx.distributed.all_sum(token, group=distributed_group)
            
            # Check for EOS
            if int(token.item()) == tokenizer.eos_token_id:
                break
        
        return {}


# Create FastAPI app
app = FastAPI(title="MLX Sharded Inference")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    global rank
    
    # Only rank 0 handles API requests
    if rank != 0:
        return ChatResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=model_name,
            choices=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
    
    try:
        # Format messages into prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        
        prompt += "Assistant: "
        
        # Generate response
        result = generate_distributed(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=model_name,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"]
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "completion_tokens": result["tokens_generated"],
                "total_tokens": len(tokenizer.encode(prompt)) + result["tokens_generated"]
            }
        )
    
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    global rank, world_size, model, shard
    
    if rank != 0:
        return {"status": "worker", "rank": rank}
    
    return {
        "status": "healthy",
        "model": model_name,
        "world_size": world_size,
        "rank": rank,
        "shard": {
            "start_layer": shard.start_layer,
            "end_layer": shard.end_layer,
            "total_layers": shard.n_layers
        } if shard else None,
        "memory": {
            "gpu_gb": round(mx.get_active_memory() / (1024**3), 2)
        }
    }


@app.get("/")
async def root():
    """Root endpoint with simple UI"""
    if rank != 0:
        return {"message": f"Worker node rank {rank}"}
    
    html = """
    <html>
        <head>
            <title>MLX Sharded Inference</title>
            <style>
                body { font-family: Arial; margin: 40px; background: #f0f0f0; }
                .container { background: white; padding: 20px; border-radius: 10px; }
                h1 { color: #333; }
                .info { background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0; }
                .endpoint { background: #f8f8f8; padding: 10px; margin: 10px 0; border-left: 3px solid #4CAF50; }
                code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 MLX Sharded Inference Server</h1>
                <div class="info">
                    <strong>Status:</strong> ✅ Running<br>
                    <strong>Model:</strong> """ + model_name + """<br>
                    <strong>Devices:</strong> """ + str(world_size) + """ GPUs<br>
                    <strong>Sharding:</strong> Layer-based (""" + str(TOTAL_LAYERS // world_size) + """ layers per GPU)
                </div>
                
                <h2>API Endpoints</h2>
                
                <div class="endpoint">
                    <strong>POST /v1/chat/completions</strong><br>
                    OpenAI-compatible chat endpoint<br>
                    <code>curl -X POST http://localhost:8100/v1/chat/completions ...</code>
                </div>
                
                <div class="endpoint">
                    <strong>GET /health</strong><br>
                    Health check and system info<br>
                    <code>curl http://localhost:8100/health</code>
                </div>
                
                <h2>Test Command</h2>
                <pre>
curl -X POST http://localhost:8100/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'
                </pre>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


def main():
    """Main entry point"""
    global rank
    
    # Load the model with sharding
    load_model_sharded()
    
    if rank == 0:
        # Only rank 0 runs the API server
        logger.info("Starting API server on http://0.0.0.0:8100")
        uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
    else:
        # Worker ranks just process requests
        logger.info(f"Worker rank {rank} standing by for distributed operations")
        
        # Keep worker alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"Worker rank {rank} shutting down")


if __name__ == "__main__":
    main()