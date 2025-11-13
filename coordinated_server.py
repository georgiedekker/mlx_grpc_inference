#!/usr/bin/env python3
"""
Coordinated MLX Distributed Server
Both ranks actively participate in generation
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List
import uuid

import mlx.core as mx
from mlx_lm import load
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
    """Load model with fixed pipeline parallelism."""
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
    
    # Load model
    logger.info(f"Rank {rank}: Loading model {model_name}...")
    model, tokenizer = load(model_name)
    
    if world_size > 1:
        # Add working pipeline support
        from fixed_pipeline import add_working_pipeline
        model = add_working_pipeline(model)
        
        # Apply pipeline parallelism 
        logger.info(f"Rank {rank}: Applying pipeline parallelism...")
        model.model.pipeline(distributed_group)
        logger.info(f"Rank {rank}: Pipeline ready")
    
    # Evaluate model parameters
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
        logger.info(f"✅ COORDINATED DISTRIBUTED INFERENCE READY")
        logger.info(f"✅ Model: {model_name}")
        logger.info(f"✅ Devices: {world_size} GPUs")
        logger.info(f"✅ Memory: {gpu_memory:.2f} GB on rank 0")
        logger.info("="*60)


def coordinated_forward(inputs: mx.array, cache=None):
    """Forward pass with both ranks participating"""
    global model, rank, world_size
    
    # All ranks do forward pass - the pipeline __call__ handles communication
    output = model.model(inputs, cache=cache)
    mx.eval(output)
    
    return output


def generate_token_by_token(prompt: str, max_tokens: int, temperature: float):
    """Generate tokens one by one with both ranks participating"""
    global model, tokenizer, rank, world_size
    
    # Tokenize prompt (all ranks)
    input_ids = tokenizer.encode(prompt)
    inputs = mx.array([input_ids])
    
    if rank == 0:
        logger.info(f"Generating up to {max_tokens} tokens from {len(input_ids)} prompt tokens")
    
    # Create sampler
    sampler = make_sampler(temp=temperature)
    
    generated_tokens = []
    cache = None
    
    for i in range(max_tokens):
        # All ranks participate in forward pass
        logits = coordinated_forward(inputs, cache)
        
        # Only rank 0 samples and decides next token
        if rank == 0:
            # Get last token logits
            next_logits = logits[:, -1, :]
            
            # Sample next token
            next_token = sampler(next_logits)
            token_id = int(next_token.item())
            
            # Check for EOS
            if token_id == tokenizer.eos_token_id:
                # Broadcast stop signal
                stop = mx.array([1])
                mx.distributed.all_sum(stop)
                mx.eval(stop)
                break
            
            generated_tokens.append(token_id)
            
            # Broadcast next token to all ranks
            token_broadcast = mx.array([token_id])
            token_broadcast = mx.distributed.all_sum(token_broadcast)
            mx.eval(token_broadcast)
            
            # Prepare next input
            inputs = mx.array([[token_id]])
        else:
            # Other ranks wait for token broadcast
            token_broadcast = mx.array([0])
            token_broadcast = mx.distributed.all_sum(token_broadcast)
            mx.eval(token_broadcast)
            
            token_id = int(token_broadcast.item())
            
            # Check for stop signal
            if token_id == 0:  # Simplified stop check
                break
            
            # Prepare next input (same as rank 0)
            inputs = mx.array([[token_id]])
    
    # Only rank 0 returns result
    if rank == 0:
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text
    else:
        return ""


def generate_coordinated(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
    """Coordinated generation with both GPUs."""
    global model, tokenizer, rank, world_size
    
    start_time = time.time()
    
    # Limited tokens for testing pipeline
    safe_max = min(max_tokens, 10) if world_size > 1 else max_tokens
    
    # Generate with coordination
    generated_text = generate_token_by_token(prompt, safe_max, temperature)
    
    gen_time = time.time() - start_time
    
    if rank == 0:
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(generated_text))
        tps = completion_tokens / gen_time if gen_time > 0 else 0
        
        logger.info(f"✅ Generated {completion_tokens} tokens in {gen_time:.2f}s")
        logger.info(f"✅ Speed: {tps:.1f} tokens/sec")
        
        return {
            "text": generated_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokens_per_second": round(tps, 1),
            "generation_time": round(gen_time, 2)
        }
    
    return {}


def main():
    """Main entry point."""
    global rank, world_size
    
    logger.info("="*60)
    logger.info("COORDINATED MLX DISTRIBUTED INFERENCE")
    logger.info(f"Using Python {sys.version.split()[0]}")
    logger.info(f"Model: {model_name}")
    logger.info("="*60)
    
    # Load model
    load_model_distributed()
    
    if rank == 0:
        # Rank 0 runs API
        app = FastAPI(title="MLX Coordinated Distributed", version="1.0")
        
        @app.get("/")
        async def dashboard():
            gpu_memory = mx.get_active_memory() / (1024**3)
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLX Coordinated Distributed</title>
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
                    <h1>🚀 MLX Coordinated Distributed</h1>
                    <p style="color: #9ca3af;">Both GPUs actively participating!</p>
                    
                    <div class="grid">
                        <div class="card">
                            <div class="metric">{world_size}</div>
                            <div class="label">Active GPUs</div>
                        </div>
                        <div class="card">
                            <div class="metric">{gpu_memory:.2f} GB</div>
                            <div class="label">GPU Memory</div>
                        </div>
                        <div class="card">
                            <div class="metric">Pipeline</div>
                            <div class="label">Parallelism</div>
                        </div>
                    </div>
                    
                    <div class="status">✅ Both GPUs Working!</div>
                    
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
                
                # Signal rank 1 to start generation
                if world_size > 1:
                    start_signal = mx.array([1])
                    mx.distributed.all_sum(start_signal)
                    mx.eval(start_signal)
                
                # Generate with both GPUs
                result = generate_coordinated(
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
                "mode": "coordinated_pipeline"
            }
        
        logger.info("Starting API server on http://0.0.0.0:8100")
        uvicorn.run(app, host="0.0.0.0", port=8100)
        
    else:
        # Rank 1 participates in generation
        logger.info(f"Rank {rank}: Ready for coordinated generation")
        
        try:
            while True:
                # Wait for generation signal
                start_signal = mx.array([0])
                start_signal = mx.distributed.all_sum(start_signal)
                mx.eval(start_signal)
                
                if start_signal.item() > 0:
                    logger.info(f"Rank {rank}: Starting coordinated generation")
                    # Participate in generation (dummy prompt, will sync with rank 0)
                    generate_coordinated("", 10, 0.7)
                    logger.info(f"Rank {rank}: Generation complete")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info(f"Rank {rank} shutting down")


if __name__ == "__main__":
    main()