#!/usr/bin/env python3
"""
Optimized MLX Distributed Inference Server
Mimics DeepSeek's pipeline approach with minimal synchronization
"""
import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import stream_generate
from mlx_lm.utils import load as mlx_load
from mlx_lm.utils import load_tokenizer
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


def add_optimized_pipeline(model_class):
    """
    Add optimized pipeline to Qwen3 - mimics DeepSeek's approach.
    Key insight: Only synchronize ONCE at the end, not after every layer!
    """
    
    # Save original forward
    if not hasattr(model_class, '_original_forward'):
        model_class._original_forward = model_class.__call__
    
    def pipeline(self, group):
        """
        Setup pipeline parallelism - similar to DeepSeek.
        Split layers in reverse: rank 0 gets last layers, last rank gets first layers.
        """
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        self.group = group
        
        if not hasattr(self, 'layers'):
            logger.warning(f"Model has no layers attribute")
            return
        
        total_layers = len(self.layers)
        layers_per_rank = total_layers // self.pipeline_size
        extra = total_layers % self.pipeline_size
        
        # Reverse distribution like DeepSeek
        if self.pipeline_rank < extra:
            layers_per_rank += 1
        
        # Calculate layer indices (reversed like DeepSeek)
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        self.num_layers = layers_per_rank
        
        logger.info(f"Rank {self.pipeline_rank}: Layers {self.start_idx}-{self.end_idx-1}")
        
        # Keep full layer list but set non-owned to None
        new_layers = self.layers[:self.end_idx]
        new_layers[:self.start_idx] = [None] * self.start_idx
        self.layers = new_layers
        self.total_layers = total_layers
        
        # Store dimensions
        if hasattr(self, 'hidden_size'):
            self.hidden_dim = self.hidden_size
        else:
            self.hidden_dim = 2048  # Qwen3-1.7B default
        
        # Force evaluation to drop unused layers
        mx.eval(self.parameters())
        
        gpu_memory = mx.get_active_memory() / (1024**3)
        logger.info(f"Rank {self.pipeline_rank}: GPU memory = {gpu_memory:.2f} GB")
    
    def optimized_forward(self, inputs, mask=None, cache=None):
        """
        Optimized forward pass - exactly like DeepSeek!
        Only ONE all_gather at the very end.
        """
        if not hasattr(self, 'pipeline_rank'):
            # Not in pipeline mode
            return self._original_forward(inputs, mask, cache)
        
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        
        # Handle embeddings (only first stage in reverse pipeline)
        if pipeline_rank == pipeline_size - 1:
            # Last rank does embeddings (owns first layers)
            if hasattr(self, 'embed_tokens'):
                h = self.embed_tokens(inputs)
            else:
                h = inputs
        else:
            # Create dummy tensor with correct shape
            if inputs.ndim == 1:
                batch_size = 1
                seq_len = inputs.shape[0]
            else:
                batch_size = inputs.shape[0] if inputs.ndim > 1 else 1
                seq_len = inputs.shape[-1]
            
            if hasattr(self, 'embed_tokens'):
                # Need embedded shape
                h = mx.zeros((batch_size, seq_len, self.hidden_dim))
            else:
                h = mx.zeros_like(inputs)
        
        # Setup cache for our layers only
        if cache is None:
            cache = [None] * self.num_layers
        
        # CRITICAL: Receive from NEXT rank (reverse pipeline)
        if pipeline_rank < pipeline_size - 1:
            logger.debug(f"Rank {pipeline_rank}: Receiving from rank {pipeline_rank + 1}")
            h = mx.distributed.recv_like(h, src=(pipeline_rank + 1))
            mx.eval(h)
        
        # Process our layers
        for i in range(self.num_layers):
            layer = self.layers[self.start_idx + i]
            if layer is not None:
                if cache[i] is not None:
                    h = layer(h, mask, cache[i])
                else:
                    h = layer(h, mask)
        
        # CRITICAL: Send to PREVIOUS rank (reverse pipeline)
        if pipeline_rank != 0:
            logger.debug(f"Rank {pipeline_rank}: Sending to rank {pipeline_rank - 1}")
            h = mx.distributed.send(h, dst=(pipeline_rank - 1))
            mx.eval(h)
        
        # Apply final norm (only rank 0 which has last layers)
        if pipeline_rank == 0:
            if hasattr(self, 'norm'):
                h = self.norm(h)
        
        # CRITICAL: Only ONE all_gather at the very end!
        # This broadcasts the final result to all ranks
        h = mx.distributed.all_gather(h)[:h.shape[0]]
        mx.eval(h)
        
        return h
    
    # Add methods
    model_class.pipeline = pipeline
    model_class.__call__ = optimized_forward
    
    logger.info(f"✅ Added optimized pipeline to {model_class.__name__}")
    return model_class


def load_and_shard_model():
    """Load cached Qwen3 model with optimized pipeline."""
    global model, tokenizer, distributed_group, rank, world_size
    
    # Initialize distributed
    distributed_group = mx.distributed.init()
    
    if not distributed_group:
        logger.info("No distributed group, loading in single GPU mode")
        model, tokenizer = mlx_load(model_name)
        rank = 0
        world_size = 1
        return
    
    rank = distributed_group.rank()
    world_size = distributed_group.size()
    hostname = os.uname().nodename
    
    logger.info(f"🚀 Rank {rank}/{world_size} on {hostname}")
    
    # Load from cache (not lazy)
    logger.info(f"Rank {rank}: Loading cached model {model_name}...")
    model, tokenizer = mlx_load(model_name, lazy=False)
    
    # Apply optimized pipeline
    if world_size > 1:
        logger.info(f"Rank {rank}: Applying optimized pipeline...")
        
        if hasattr(model, 'model'):
            # Qwen3 has model.model structure
            inner_model = model.model
            add_optimized_pipeline(type(inner_model))
            inner_model.pipeline(distributed_group)
        else:
            add_optimized_pipeline(type(model))
            model.pipeline(distributed_group)
    
    # Synchronize
    if world_size > 1:
        logger.info(f"Rank {rank}: Synchronizing...")
        sync = mx.distributed.all_sum(mx.array([1.0]))
        mx.eval(sync)
        logger.info(f"Rank {rank}: Ready")
    
    if rank == 0:
        gpu_memory = mx.get_active_memory() / (1024**3)
        logger.info("="*60)
        logger.info(f"✅ OPTIMIZED DISTRIBUTED INFERENCE READY")
        logger.info(f"✅ Model: {model_name} (cached)")
        logger.info(f"✅ Devices: {world_size} GPUs")
        logger.info(f"✅ Pipeline: Optimized (single all_gather)")
        logger.info(f"✅ Memory: {gpu_memory:.2f} GB on rank 0")
        logger.info("="*60)


def generate_distributed(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
    """Generate text using optimized pipeline."""
    global model, tokenizer, rank, world_size
    
    start_time = time.time()
    
    # Only rank 0 handles prompt
    if rank == 0:
        prompt_tokens = len(tokenizer.encode(prompt))
        logger.info(f"Generating {max_tokens} tokens from {prompt_tokens} prompt tokens")
    else:
        prompt_tokens = 0
        prompt = ""
    
    # Create sampler
    sampler = make_sampler(temp=temperature)
    
    # All ranks participate
    responses = []
    generated_text = ""
    
    for response in stream_generate(
        model,
        tokenizer,
        prompt if rank == 0 else "",
        max_tokens=max_tokens,
        sampler=sampler
    ):
        if rank == 0:
            responses.append(response)
            if hasattr(response, 'text'):
                generated_text = response.text
    
    gen_time = time.time() - start_time
    
    # Only rank 0 returns results
    if rank == 0 and responses:
        final = responses[-1] if responses else None
        
        if final:
            completion_tokens = final.generation_tokens if hasattr(final, 'generation_tokens') else len(tokenizer.encode(generated_text)) - prompt_tokens
            gen_tps = final.generation_tps if hasattr(final, 'generation_tps') else completion_tokens / gen_time
            
            logger.info(f"✅ Generated {completion_tokens} tokens in {gen_time:.2f}s")
            logger.info(f"✅ Speed: {gen_tps:.1f} tokens/sec with {world_size} GPUs")
            
            return {
                "text": generated_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "tokens_per_second": round(gen_tps, 1),
                "generation_time": round(gen_time, 2),
                "gpus_used": world_size
            }
    
    return {}


def main():
    """Main entry point."""
    global rank, world_size
    
    logger.info("="*60)
    logger.info("OPTIMIZED MLX DISTRIBUTED INFERENCE")
    logger.info(f"Using uv + Python {sys.version.split()[0]}")
    logger.info(f"Model: {model_name} (cached)")
    logger.info("="*60)
    
    # Load and shard
    load_and_shard_model()
    
    # Only rank 0 runs API
    if rank == 0:
        app = FastAPI(title="MLX Optimized Pipeline", version="1.0")
        
        @app.get("/")
        async def dashboard():
            gpu_memory = mx.get_active_memory() / (1024**3)
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLX Optimized Pipeline</title>
                <style>
                    body {{ font-family: -apple-system, sans-serif; background: #0a0a0a; color: #fff; padding: 40px; }}
                    .container {{ max-width: 900px; margin: 0 auto; }}
                    h1 {{ color: #10b981; font-size: 2.5em; }}
                    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
                    .card {{ background: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #333; }}
                    .metric {{ color: #60a5fa; font-size: 2em; font-weight: bold; }}
                    .label {{ color: #9ca3af; font-size: 0.9em; margin-top: 5px; }}
                    .status {{ background: #10b981; color: #000; padding: 10px 20px; border-radius: 20px; display: inline-block; font-weight: 600; }}
                    .warning {{ background: #f59e0b; color: #000; padding: 10px; border-radius: 8px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 MLX Optimized Pipeline</h1>
                    <p style="color: #9ca3af;">Single all_gather synchronization (like DeepSeek)</p>
                    
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
                            <div class="metric">Optimized</div>
                            <div class="label">Pipeline Type</div>
                        </div>
                    </div>
                    
                    <div class="warning">
                        ⚠️ Note: This uses send/recv which may deadlock in MLX
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
                        "generation_time": result.get("generation_time", 0),
                        "gpus_used": result.get("gpus_used", 1)
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
                "pipeline": "optimized_single_sync",
                "python": sys.version.split()[0]
            }
        
        logger.info("Starting API server on http://0.0.0.0:8100")
        uvicorn.run(app, host="0.0.0.0", port=8100)
    else:
        # Worker ranks
        logger.info(f"Worker rank {rank} ready")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"Worker rank {rank} shutting down")


if __name__ == "__main__":
    main()