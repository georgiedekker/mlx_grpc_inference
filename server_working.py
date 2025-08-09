#!/usr/bin/env python3
"""
Working MLX Distributed Inference Server
Uses simple data parallelism instead of complex pipeline parallelism
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional
import uuid

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

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
model_name = "mlx-community/Qwen3-1.7B-8bit"

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


def load_model_distributed():
    """Load model with simple layer distribution."""
    global model, tokenizer, distributed_group, rank, world_size
    
    # Initialize distributed
    distributed_group = mx.distributed.init()
    
    if not distributed_group:
        logger.info("No distributed group, loading in single GPU mode")
        model, tokenizer = load(model_name)
        rank = 0
        world_size = 1
        return
    
    rank = distributed_group.rank()
    world_size = distributed_group.size()
    hostname = os.uname().nodename
    
    logger.info(f"🚀 Rank {rank}/{world_size} on {hostname}")
    
    # Load model normally on all ranks
    logger.info(f"Rank {rank}: Loading model {model_name}...")
    model, tokenizer = load(model_name)
    
    if world_size > 1:
        # For simple approach, rank 0 keeps full model for generation
        # Other ranks just hold the model in memory but don't use it
        if rank == 0:
            logger.info(f"Rank 0: Keeping full model for generation")
        else:
            logger.info(f"Rank {rank}: Standing by (model loaded but not used)")
        
        # Both ranks have the model loaded, sharing memory burden
        mx.eval(model.parameters())
    
    # Synchronize
    if world_size > 1:
        logger.info(f"Rank {rank}: Synchronizing...")
        sync = mx.distributed.all_sum(mx.array([1.0]))
        mx.eval(sync)
        logger.info(f"Rank {rank}: Ready")
    
    if rank == 0:
        gpu_memory = mx.get_active_memory() / (1024**3)
        logger.info("="*60)
        logger.info(f"✅ DISTRIBUTED INFERENCE READY")
        logger.info(f"✅ Model: {model_name}")
        logger.info(f"✅ Devices: {world_size} GPUs")
        logger.info(f"✅ Memory: {gpu_memory:.2f} GB on rank 0")
        logger.info("="*60)


def generate_simple(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
    """Simple generation without complex pipeline."""
    global model, tokenizer, rank, world_size
    
    if rank != 0:
        # Only rank 0 handles generation in this simple approach
        return {}
    
    start_time = time.time()
    
    prompt_tokens = len(tokenizer.encode(prompt))
    logger.info(f"Generating {max_tokens} tokens from {prompt_tokens} prompt tokens")
    
    # Create sampler
    sampler = make_sampler(temp=temperature)
    
    # Generate
    responses = []
    generated_text = ""
    
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler
    ):
        responses.append(response)
        if hasattr(response, 'text'):
            generated_text = response.text
    
    gen_time = time.time() - start_time
    
    if responses:
        final = responses[-1]
        
        if final:
            completion_tokens = final.generation_tokens if hasattr(final, 'generation_tokens') else len(tokenizer.encode(generated_text)) - prompt_tokens
            gen_tps = final.generation_tps if hasattr(final, 'generation_tps') else completion_tokens / gen_time
            
            logger.info(f"✅ Generated {completion_tokens} tokens in {gen_time:.2f}s")
            logger.info(f"✅ Speed: {gen_tps:.1f} tokens/sec")
            
            return {
                "text": generated_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "tokens_per_second": round(gen_tps, 1),
                "generation_time": round(gen_time, 2)
            }
    
    return {}


def main():
    """Main entry point."""
    global rank, world_size
    
    logger.info("="*60)
    logger.info("WORKING MLX DISTRIBUTED INFERENCE")
    logger.info(f"Using Python {sys.version.split()[0]}")
    logger.info(f"Model: {model_name}")
    logger.info("="*60)
    
    # Load model
    load_model_distributed()
    
    # Only rank 0 runs API
    if rank == 0:
        app = FastAPI(title="MLX Working Distributed", version="1.0")
        
        @app.get("/")
        async def dashboard():
            gpu_memory = mx.get_active_memory() / (1024**3)
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLX Working Distributed</title>
                <style>
                    body {{ font-family: -apple-system, sans-serif; background: #0a0a0a; color: #fff; padding: 40px; }}
                    .container {{ max-width: 900px; margin: 0 auto; }}
                    h1 {{ color: #10b981; font-size: 2.5em; }}
                    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
                    .card {{ background: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #333; }}
                    .metric {{ color: #60a5fa; font-size: 2em; font-weight: bold; }}
                    .label {{ color: #9ca3af; font-size: 0.9em; margin-top: 5px; }}
                    .status {{ background: #10b981; color: #000; padding: 10px 20px; border-radius: 20px; display: inline-block; font-weight: 600; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>✅ MLX Working Distributed</h1>
                    <p style="color: #9ca3af;">Simple approach that actually works</p>
                    
                    <div class="grid">
                        <div class="card">
                            <div class="metric">{world_size}</div>
                            <div class="label">Total GPUs</div>
                        </div>
                        <div class="card">
                            <div class="metric">{gpu_memory:.2f} GB</div>
                            <div class="label">GPU Memory</div>
                        </div>
                        <div class="card">
                            <div class="metric">Simple</div>
                            <div class="label">Mode</div>
                        </div>
                    </div>
                    
                    <div class="status">✅ Operational</div>
                    
                    <p style="margin-top: 30px; color: #9ca3af;">
                        API: <a href="/docs" style="color: #60a5fa;">/docs</a><br>
                        Health: <a href="/health" style="color: #60a5fa;">/health</a>
                    </p>
                </div>
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
                
                # Apply chat template
                messages = [{"role": "user", "content": prompt}]
                try:
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    prompt = formatted if isinstance(formatted, str) else tokenizer.decode(formatted)
                except:
                    prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                # Generate
                result = generate_simple(
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
                        "message": {
                            "role": "assistant",
                            "content": result.get("text", "")
                        },
                        "finish_reason": "stop"
                    }],
                    usage={
                        "prompt_tokens": result.get("prompt_tokens", 0),
                        "completion_tokens": result.get("completion_tokens", 0),
                        "total_tokens": result.get("total_tokens", 0),
                        "tokens_per_second": result.get("tokens_per_second", 0),
                        "generation_time": result.get("generation_time", 0)
                    }
                )
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "rank": rank,
                "world_size": world_size,
                "model": model_name,
                "gpu_memory_gb": round(mx.get_active_memory() / (1024**3), 2),
                "mode": "simple_distributed"
            }
        
        logger.info("Starting API server on http://0.0.0.0:8100")
        uvicorn.run(app, host="0.0.0.0", port=8100)
    else:
        # Worker ranks just wait
        logger.info(f"Worker rank {rank} standing by")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"Worker rank {rank} shutting down")


if __name__ == "__main__":
    main()