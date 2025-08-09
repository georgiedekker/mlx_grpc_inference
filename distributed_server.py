#!/usr/bin/env python3
"""
MLX Distributed Inference Server with Manual Sharding
Uses send/recv for communication between ranks like exo
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
from mlx_lm.models.cache import make_prompt_cache

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
group = None
rank = 0
world_size = 1
shard = None
model_name = "mlx-community/Qwen3-1.7B-8bit"
cache = None

# Model configuration
TOTAL_LAYERS = 28  # Qwen3-1.7B has 28 layers


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7


def get_model_path(model_name: str) -> Path:
    """Get the local path for a model"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_id = f"models--{model_name.replace('/', '--')}"
    model_dir = cache_dir / model_id
    
    if not model_dir.exists():
        return None
    
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
        if snapshot_dirs:
            return snapshot_dirs[-1]
    
    return None


def load_model_distributed():
    """Load model with manual sharding across devices"""
    global model, tokenizer, group, rank, world_size, shard, cache
    
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        logger.error("No distributed group found. Please run with mpirun")
        sys.exit(1)
    
    rank = group.rank()
    world_size = group.size()
    hostname = os.uname().nodename
    
    logger.info(f"🚀 Rank {rank}/{world_size} on {hostname}")
    
    if world_size != 2:
        logger.error(f"This script requires exactly 2 GPUs, got {world_size}")
        sys.exit(1)
    
    # Get model path
    model_path = get_model_path(model_name)
    if not model_path:
        # For simplicity, just load the model normally to get path
        logger.info(f"Downloading/loading model {model_name}...")
        temp_model, _ = load(model_name)
        del temp_model
        mx.eval(mx.array([0]))  # Clear memory
        model_path = get_model_path(model_name)
    
    logger.info(f"Model path: {model_path}")
    
    # Define shards for each rank
    layers_per_device = TOTAL_LAYERS // world_size
    
    if rank == 0:
        # First device: layers 0-13 + embeddings
        shard = Shard(
            model_id=model_name,
            start_layer=0,
            end_layer=layers_per_device - 1,
            n_layers=TOTAL_LAYERS
        )
        logger.info(f"Rank 0: Loading layers {shard.start_layer}-{shard.end_layer} (first shard with embeddings)")
    else:
        # Second device: layers 14-27 + LM head
        shard = Shard(
            model_id=model_name,
            start_layer=layers_per_device,
            end_layer=TOTAL_LAYERS - 1,
            n_layers=TOTAL_LAYERS
        )
        logger.info(f"Rank 1: Loading layers {shard.start_layer}-{shard.end_layer} (last shard with LM head)")
    
    # Load sharded model
    logger.info(f"Rank {rank}: Loading sharded model...")
    model = load_sharded_model(str(model_path), shard, lazy=False)
    
    # Load tokenizer (both ranks need it)
    tokenizer = load_tokenizer(model_path)
    
    # Create cache
    cache = make_prompt_cache(model)
    
    # Log memory usage
    gpu_memory = mx.get_active_memory() / (1024**3)
    logger.info(f"Rank {rank}: GPU memory used: {gpu_memory:.2f} GB")
    
    # Synchronize
    logger.info(f"Rank {rank}: Synchronizing...")
    sync = mx.distributed.all_sum(mx.array([1.0], dtype=mx.float32), group=group)
    mx.eval(sync)
    logger.info(f"Rank {rank}: Ready")
    
    if rank == 0:
        logger.info("="*60)
        logger.info("✅ DISTRIBUTED SHARDED INFERENCE READY")
        logger.info(f"✅ Model: {model_name}")
        logger.info(f"✅ Devices: {world_size} GPUs")
        logger.info(f"✅ Rank 0: Layers 0-{layers_per_device-1} + embeddings")
        logger.info(f"✅ Rank 1: Layers {layers_per_device}-{TOTAL_LAYERS-1} + LM head")
        logger.info("="*60)


def distributed_forward(inputs: mx.array, use_cache=None) -> mx.array:
    """Forward pass through distributed sharded model"""
    global model, rank, world_size, group, shard
    
    # Rank 0: Process first shard
    if rank == 0:
        # Process through our layers (includes embeddings)
        output = model(inputs, use_cache)
        mx.eval(output)
        
        # Send output shape first, then data
        output_shape = mx.array(output.shape, dtype=mx.int32)
        mx.distributed.send(output_shape, dst=1, group=group)
        mx.eval(output_shape)
        
        # Flatten and send the actual data
        output_flat = output.reshape(-1)
        mx.distributed.send(output_flat, dst=1, group=group)
        mx.eval(output_flat)
        
        # Rank 0 waits for final result from rank 1
        # Receive shape first
        result_shape = mx.distributed.recv_like(mx.array([0, 0, 0], dtype=mx.int32), src=1, group=group)
        mx.eval(result_shape)
        
        # Receive the actual logits
        result_size = int(result_shape[0] * result_shape[1] * result_shape[2])
        result_flat = mx.distributed.recv_like(mx.zeros(result_size, dtype=mx.float32), src=1, group=group)
        mx.eval(result_flat)
        
        # Reshape to original
        result = result_flat.reshape(tuple(result_shape.tolist()))
        return result
    
    # Rank 1: Receive from rank 0 and process second shard
    else:
        # Receive shape first
        input_shape = mx.distributed.recv_like(mx.array([0, 0, 0], dtype=mx.int32), src=0, group=group)
        mx.eval(input_shape)
        
        # Receive the actual data
        input_size = int(input_shape[0] * input_shape[1] * input_shape[2])
        input_flat = mx.distributed.recv_like(mx.zeros(input_size, dtype=mx.float32), src=0, group=group)
        mx.eval(input_flat)
        
        # Reshape to original
        received = input_flat.reshape(tuple(input_shape.tolist()))
        
        # Process through our layers (includes LM head)
        output = model(received, use_cache)
        mx.eval(output)
        
        # Send result back to rank 0
        output_shape = mx.array(output.shape, dtype=mx.int32)
        mx.distributed.send(output_shape, dst=0, group=group)
        mx.eval(output_shape)
        
        output_flat = output.reshape(-1)
        mx.distributed.send(output_flat, dst=0, group=group)
        mx.eval(output_flat)
        
        return output


def generate_distributed(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
    """Generate text using distributed sharded model"""
    global model, tokenizer, rank, world_size, cache
    
    # All ranks participate
    # Tokenize prompt (all ranks do this for simplicity)
    tokens = tokenizer.encode(prompt)
    prompt_tokens = mx.array(tokens).reshape(1, -1)
    
    if rank == 0:
        logger.info(f"Generating {max_tokens} tokens from {len(tokens)} prompt tokens")
    
    start_time = time.time()
    generated_tokens = []
    
    # Create sampler
    sampler = make_sampler(temperature)
    
    # Process prompt through model
    logits = distributed_forward(prompt_tokens, cache)
    
    # Generate tokens
    for i in range(max_tokens):
        # Only rank 0 samples (since it has the final logits)
        if rank == 0:
            # Sample next token
            logits_last = logits[:, -1, :]
            token = sampler(logits_last)
            mx.eval(token)
            token_id = int(token.item())
            
            generated_tokens.append(token_id)
            
            # Check for EOS
            if token_id == tokenizer.eos_token_id:
                # Send stop signal to rank 1
                stop_signal = mx.array([-1], dtype=mx.int32)
                mx.distributed.send(stop_signal, dst=1, group=group)
                mx.eval(stop_signal)
                break
            
            # Send token to rank 1 for next iteration
            token_array = mx.array([token_id], dtype=mx.int32)
            mx.distributed.send(token_array, dst=1, group=group)
            mx.eval(token_array)
            
            # Prepare next input
            next_input = mx.array([[token_id]])
            logits = distributed_forward(next_input, cache)
        else:
            # Rank 1: receive token and check for stop
            token_array = mx.distributed.recv_like(mx.array([0], dtype=mx.int32), src=0, group=group)
            mx.eval(token_array)
            token_id = int(token_array.item())
            
            if token_id == -1:  # Stop signal
                break
            
            # Participate in forward pass
            next_input = mx.array([[token_id]])
            _ = distributed_forward(next_input, cache)
    
    # Calculate metrics (rank 0 only)
    if rank == 0:
        total_time = time.time() - start_time
        tokens_per_second = len(generated_tokens) / total_time if total_time > 0 else 0
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens)
        
        logger.info(f"✅ Generated {len(generated_tokens)} tokens in {total_time:.2f}s")
        logger.info(f"✅ Speed: {tokens_per_second:.1f} tokens/sec")
        
        return {
            "text": generated_text,
            "tokens_generated": len(generated_tokens),
            "tokens_per_second": tokens_per_second,
            "time_taken": total_time,
            "prompt_tokens": len(tokens)
        }
    else:
        return {}


# Create FastAPI app
app = FastAPI(title="MLX Distributed Sharded Inference")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    global rank, model_name
    
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
    
    # Generate response (all ranks participate)
    result = generate_distributed(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    # Only rank 0 returns the API response
    if rank == 0:
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["tokens_generated"],
                "total_tokens": result["prompt_tokens"] + result["tokens_generated"]
            }
        }
    else:
        # Rank 1 returns minimal response
        return {"status": "processed"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    global rank, world_size, model_name, shard
    
    return {
        "status": "healthy",
        "model": model_name,
        "world_size": world_size,
        "rank": rank,
        "hostname": os.uname().nodename,
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
    global rank, world_size, model_name
    
    if rank != 0:
        return {"message": f"Worker node rank {rank}"}
    
    html = f"""
    <html>
        <head>
            <title>MLX Distributed Sharded Inference</title>
            <style>
                body {{ font-family: Arial; margin: 40px; background: #f0f0f0; }}
                .container {{ background: white; padding: 20px; border-radius: 10px; }}
                h1 {{ color: #333; }}
                .info {{ background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .endpoint {{ background: #f8f8f8; padding: 10px; margin: 10px 0; border-left: 3px solid #4CAF50; }}
                code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                .status {{ color: green; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 MLX Distributed Sharded Inference</h1>
                <div class="info">
                    <div class="status">✅ RUNNING</div>
                    <strong>Model:</strong> {model_name}<br>
                    <strong>Devices:</strong> {world_size} GPUs<br>
                    <strong>Mode:</strong> Manual Sharding with send/recv<br>
                    <strong>Distribution:</strong><br>
                    • Rank 0: Layers 0-13 + embeddings<br>
                    • Rank 1: Layers 14-27 + LM head
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
  -d '{{
    "messages": [{{"role": "user", "content": "What is 2+2?"}}],
    "max_tokens": 20,
    "temperature": 0.7
  }}'
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
    load_model_distributed()
    
    if rank == 0:
        # Only rank 0 runs the API server
        logger.info("Starting API server on http://0.0.0.0:8100")
        uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
    else:
        # Rank 1 participates in distributed operations
        logger.info(f"Worker rank {rank} running in distributed mode")
        
        # Keep worker alive and handle requests
        try:
            # Also run the API server on rank 1 to handle distributed requests
            uvicorn.run(app, host="0.0.0.0", port=8101, log_level="warning")
        except KeyboardInterrupt:
            logger.info(f"Worker rank {rank} shutting down")


if __name__ == "__main__":
    main()