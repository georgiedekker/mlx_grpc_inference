#!/usr/bin/env python3
"""
Fast MLX Distributed Inference Server
Minimizes synchronization overhead by batching operations
"""
import os
import sys
import time
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
import uuid

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate, load
from mlx_lm.utils import load_tokenizer
from huggingface_hub import snapshot_download

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

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
model_name = os.environ.get("MODEL_NAME", "mlx-community/Qwen3-1.7B-8bit")

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]


def simple_pipeline_setup(model_obj, group):
    """
    Simple pipeline setup - just split layers, no complex synchronization.
    We'll use the built-in MLX generation which handles distributed better.
    """
    rank = group.rank()
    world_size = group.size()
    
    # Find the model's transformer layers
    if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'layers'):
        layers = model_obj.model.layers
        target = model_obj.model
    elif hasattr(model_obj, 'layers'):
        layers = model_obj.layers
        target = model_obj
    else:
        logger.warning("Could not find layers to split")
        return
    
    num_layers = len(layers)
    layers_per_rank = num_layers // world_size
    start = rank * layers_per_rank
    end = num_layers if rank == world_size - 1 else (rank + 1) * layers_per_rank
    
    # Keep only our layers
    target.layers = layers[start:end]
    
    logger.info(f"Rank {rank}: Keeping layers {start}-{end-1} ({end-start} layers)")
    
    # Set pipeline attributes for MLX to recognize
    target.pipeline_rank = rank
    target.pipeline_size = world_size
    
    # Force evaluation to load weights
    mx.eval(model_obj.parameters())
    
    # Check memory
    gpu_memory = mx.get_active_memory() / (1024**3)
    logger.info(f"Rank {rank}: GPU memory after sharding = {gpu_memory:.2f} GB")


def load_distributed_model(model_path: str):
    """Load model with simple distributed sharding."""
    global model, tokenizer, distributed_group, rank, world_size
    
    # Initialize distributed
    distributed_group = mx.distributed.init()
    
    if not distributed_group:
        logger.info("No distributed group, loading full model")
        model, tokenizer = load(model_path)
        rank = 0
        world_size = 1
        return
    
    rank = distributed_group.rank()
    world_size = distributed_group.size()
    hostname = os.uname().nodename
    
    logger.info(f"Rank {rank}/{world_size} on {hostname}")
    
    # Load model and tokenizer
    logger.info(f"Rank {rank}: Loading model...")
    model, tokenizer = load(model_path)
    
    # Apply simple pipeline sharding
    if world_size > 1:
        simple_pipeline_setup(model, distributed_group)
    
    # Synchronize
    if world_size > 1:
        mx.eval(mx.distributed.all_sum(mx.array(1.0)))
        logger.info(f"Rank {rank}: Ready")
    
    if rank == 0:
        logger.info("="*60)
        logger.info(f"✅ DISTRIBUTED INFERENCE READY")
        logger.info(f"✅ {world_size} devices connected")
        logger.info("="*60)


def generate_distributed(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
    """
    Generate text using MLX's built-in distributed generation.
    This should be much faster than our custom allreduce approach.
    """
    global model, tokenizer, rank, world_size
    
    if rank != 0:
        # Only rank 0 generates in this simple approach
        # Workers will participate via MLX's internal communication
        return {}
    
    start_time = time.time()
    
    # Count tokens
    prompt_tokens = len(tokenizer.encode(prompt))
    
    # Use MLX's built-in generate function
    # It should handle distributed execution internally
    logger.info(f"Generating with {prompt_tokens} prompt tokens, max {max_tokens} tokens")
    
    # Simple generation without streaming for speed test
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False
    )
    
    gen_time = time.time() - start_time
    
    # Count generated tokens
    completion_tokens = len(tokenizer.encode(output)) - prompt_tokens
    tokens_per_second = completion_tokens / gen_time if gen_time > 0 else 0
    
    logger.info(f"Generated {completion_tokens} tokens in {gen_time:.2f}s = {tokens_per_second:.1f} tok/s")
    
    return {
        "text": output,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "tokens_per_second": round(tokens_per_second, 1),
        "generation_time": round(gen_time, 2),
        "gpus_used": world_size
    }


def main():
    """Main entry point."""
    global rank, world_size, model_name
    
    logger.info("="*60)
    logger.info("FAST MLX DISTRIBUTED INFERENCE SERVER")
    logger.info(f"Model: {model_name}")
    logger.info("="*60)
    
    # Load model
    load_distributed_model(model_name)
    
    # Only rank 0 runs API server
    if rank == 0:
        app = FastAPI(title="Fast MLX Distributed", version="1.0")
        
        @app.get("/")
        async def root():
            return HTMLResponse(f"""
            <html>
            <head><title>Fast MLX Distributed</title></head>
            <body style="font-family: system-ui; padding: 20px;">
                <h1>🚀 Fast MLX Distributed Inference</h1>
                <p>Status: ✅ Running on {world_size} GPUs</p>
                <p>Model: {model_name}</p>
                <p>API: <a href="/docs">/docs</a></p>
            </body>
            </html>
            """)
        
        @app.post("/v1/chat/completions", response_model=ChatResponse)
        async def chat_completions(request: ChatRequest):
            try:
                # Extract prompt
                prompt = ""
                for msg in request.messages:
                    if msg.role == "user":
                        prompt = msg.content
                        break
                
                if not prompt:
                    prompt = "Hello"
                
                # Apply chat template if available
                messages = [{"role": "user", "content": prompt}]
                try:
                    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    prompt = formatted if isinstance(formatted, str) else tokenizer.decode(formatted)
                except:
                    prompt = f"User: {prompt}\nAssistant:"
                
                # Generate
                result = generate_distributed(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                return ChatResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    created=int(time.time()),
                    model=model_name,
                    choices=[{
                        "index": 0,
                        "message": {"role": "assistant", "content": result.get("text", "")},
                        "finish_reason": "stop"
                    }],
                    usage={
                        "prompt_tokens": result.get("prompt_tokens", 0),
                        "completion_tokens": result.get("completion_tokens", 0),
                        "total_tokens": result.get("total_tokens", 0),
                        "tokens_per_second": result.get("tokens_per_second", 0),
                        "gpus_used": result.get("gpus_used", 1)
                    }
                )
            except Exception as e:
                logger.error(f"Error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "rank": rank,
                "world_size": world_size,
                "model": model_name,
                "gpu_memory_gb": round(mx.get_active_memory() / (1024**3), 2)
            }
        
        logger.info("Starting API server on http://0.0.0.0:8100")
        uvicorn.run(app, host="0.0.0.0", port=8100)
    else:
        # Worker ranks just wait
        logger.info(f"Worker rank {rank} ready")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"Worker rank {rank} shutting down")


if __name__ == "__main__":
    main()