#!/usr/bin/env python3
"""
MLX Pipeline Parallel Inference Server
Proper implementation using pipeline parallelism similar to exo
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import uuid
from pathlib import Path
import json
import glob

import mlx.core as mx
import mlx.nn as nn
import mlx.distributed as dist
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache, RotatingKVCache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state
model = None
tokenizer = None
group = None
rank = 0
world_size = 1
model_name = "mlx-community/Qwen3-1.7B-8bit"
model_path = None
config = None

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


def load_model_pipeline():
    """Load model using MLX's native pipeline parallelism"""
    global model, tokenizer, group, rank, world_size, model_path, config
    
    # Initialize distributed
    group = dist.init()
    
    if not group:
        logger.error("No distributed group found. Please run with mpirun")
        sys.exit(1)
    
    rank = group.rank()
    world_size = group.size()
    hostname = os.uname().nodename
    
    logger.info(f"🚀 Rank {rank}/{world_size} on {hostname}")
    
    # Get model path
    model_path = get_model_path(model_name)
    if not model_path:
        logger.error(f"Could not find model {model_name}")
        sys.exit(1)
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Import the right model class based on config
    model_type = config.get("model_type", "qwen2")
    
    if model_type == "qwen2":
        from mlx_lm.models.qwen2 import Model, ModelArgs
    else:
        logger.error(f"Unsupported model type: {model_type}")
        sys.exit(1)
    
    # Create model args
    model_args = ModelArgs.from_dict(config)
    
    # Load the model
    logger.info(f"Rank {rank}: Loading model...")
    model = Model(model_args)
    
    # Load weights
    weight_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    
    # Apply quantization if needed
    if (quantization := config.get("quantization", None)) is not None:
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights
        
        nn.quantize(model, **quantization, class_predicate=class_predicate)
    
    # Load weights
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    
    # Setup pipeline parallelism
    # This distributes the model layers across devices
    logger.info(f"Rank {rank}: Setting up pipeline parallelism...")
    model = model.pipeline(group)
    
    # Load tokenizer (all ranks need it for now)
    tokenizer = load_tokenizer(str(model_path))
    
    # Log memory usage
    gpu_memory = mx.get_active_memory() / (1024**3)
    logger.info(f"Rank {rank}: GPU memory used: {gpu_memory:.2f} GB")
    
    # Synchronize
    logger.info(f"Rank {rank}: Synchronizing...")
    sync = dist.all_sum(mx.array([1.0]), group=group)
    mx.eval(sync)
    logger.info(f"Rank {rank}: Ready")
    
    if rank == 0:
        logger.info("="*60)
        logger.info("✅ PIPELINE PARALLEL INFERENCE READY")
        logger.info(f"✅ Model: {model_name}")
        logger.info(f"✅ Devices: {world_size} GPUs")
        logger.info(f"✅ Pipeline: {TOTAL_LAYERS} layers across {world_size} devices")
        logger.info(f"✅ ~{TOTAL_LAYERS // world_size} layers per device")
        logger.info("="*60)


def generate_pipeline(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
    """Generate text using pipeline parallel model"""
    global model, tokenizer, rank, world_size, group
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    prompt_tokens = mx.array(tokens)
    
    if rank == 0:
        logger.info(f"Generating {max_tokens} tokens from {len(tokens)} prompt tokens")
    
    start_time = time.time()
    generated_tokens = []
    
    # Create sampler
    sampler = make_sampler(temperature)
    
    # Create cache for the model
    cache = make_prompt_cache(model)
    
    # Process prompt through model
    inputs = prompt_tokens[None, :]  # Add batch dimension
    logits = model(inputs, cache)
    mx.eval(logits)
    
    # Generate tokens
    for i in range(max_tokens):
        # Sample next token (only on last rank)
        if rank == world_size - 1:
            # Sample from logits
            logits_last = logits[:, -1, :]
            token = sampler(logits_last)
            mx.eval(token)
            token_id = int(token.item())
        else:
            token_id = 0
        
        # Broadcast token to all ranks
        token_array = mx.array([token_id], dtype=mx.int32)
        token_array = dist.all_sum(token_array, group=group)
        token_id = int(token_array.item()) // world_size  # Correct for all_sum
        
        generated_tokens.append(token_id)
        
        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            break
        
        # Prepare next input
        next_input = mx.array([[token_id]])
        logits = model(next_input, cache)
        mx.eval(logits)
    
    # Calculate metrics
    total_time = time.time() - start_time
    tokens_per_second = len(generated_tokens) / total_time if total_time > 0 else 0
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens)
    
    if rank == 0:
        logger.info(f"✅ Generated {len(generated_tokens)} tokens in {total_time:.2f}s")
        logger.info(f"✅ Speed: {tokens_per_second:.1f} tokens/sec")
    
    return {
        "text": generated_text,
        "tokens_generated": len(generated_tokens),
        "tokens_per_second": tokens_per_second,
        "time_taken": total_time,
        "prompt_tokens": len(tokens)
    }


# Create FastAPI app
app = FastAPI(title="MLX Pipeline Parallel Inference")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    global rank, model_name
    
    # All ranks participate in generation due to pipeline parallelism
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
        
        # Generate response (all ranks participate)
        result = generate_pipeline(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Only rank 0 returns the response
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
            # Other ranks return empty response
            return {"status": "processed"}
    
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        if rank == 0:
            raise HTTPException(status_code=500, detail=str(e))
        else:
            return {"error": str(e)}


@app.get("/health")
async def health():
    """Health check endpoint"""
    global rank, world_size, model_name
    
    return {
        "status": "healthy",
        "model": model_name,
        "world_size": world_size,
        "rank": rank,
        "hostname": os.uname().nodename,
        "memory": {
            "gpu_gb": round(mx.get_active_memory() / (1024**3), 2)
        },
        "pipeline": True
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
            <title>MLX Pipeline Parallel Inference</title>
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
                <h1>🚀 MLX Pipeline Parallel Inference</h1>
                <div class="info">
                    <div class="status">✅ RUNNING</div>
                    <strong>Model:</strong> {model_name}<br>
                    <strong>Devices:</strong> {world_size} GPUs<br>
                    <strong>Mode:</strong> Pipeline Parallel<br>
                    <strong>Layers:</strong> {TOTAL_LAYERS} distributed across devices
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
    "messages": [{{"role": "user", "content": "Hello!"}}],
    "max_tokens": 50,
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
    
    # Load the model with pipeline parallelism
    load_model_pipeline()
    
    if rank == 0:
        # Only rank 0 runs the API server
        logger.info("Starting API server on http://0.0.0.0:8100")
        uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
    else:
        # Worker ranks participate in distributed operations
        logger.info(f"Worker rank {rank} running in pipeline mode")
        
        # Run a simple event loop to handle distributed operations
        import asyncio
        asyncio.run(serve_worker())


async def serve_worker():
    """Worker event loop for handling pipeline operations"""
    global rank
    logger.info(f"Worker rank {rank} ready for pipeline operations")
    
    # The worker needs to stay alive and participate in the FastAPI app
    # This is handled by uvicorn in a distributed manner
    import asyncio
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    main()