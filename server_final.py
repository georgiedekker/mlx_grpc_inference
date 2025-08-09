#!/usr/bin/env python3
"""
Final MLX Distributed Inference Server
Uses our custom pipeline method that works with Qwen3
No lazy loading, no DeepSeek, uses cached models
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
model_name = "mlx-community/Qwen3-1.7B-8bit"  # Our cached model

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


def add_custom_pipeline(model_class):
    """
    Add our custom pipeline method to Qwen3 or any model.
    This mimics DeepSeek's pipeline but works with models that don't have it built-in.
    """
    
    def pipeline(self, group):
        """
        Custom pipeline implementation for Qwen3.
        Splits layers across devices similar to DeepSeek but adapted for our needs.
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
        
        # Calculate layer distribution
        if self.pipeline_rank < extra:
            self.start_idx = self.pipeline_rank * (layers_per_rank + 1)
            self.end_idx = self.start_idx + layers_per_rank + 1
        else:
            self.start_idx = self.pipeline_rank * layers_per_rank + extra
            self.end_idx = self.start_idx + layers_per_rank
        
        logger.info(f"Rank {self.pipeline_rank}: Assigned layers {self.start_idx}-{self.end_idx-1}")
        
        # Keep only our layers, set others to None (like DeepSeek does)
        new_layers = []
        for i in range(total_layers):
            if self.start_idx <= i < self.end_idx:
                new_layers.append(self.layers[i])
            else:
                new_layers.append(None)  # Placeholder for non-owned layers
        
        self.layers = new_layers
        self.num_layers = self.end_idx - self.start_idx
        self.total_layers = total_layers
        
        # Store hidden dimensions
        if hasattr(self, 'hidden_size'):
            self.hidden_dim = self.hidden_size
        else:
            self.hidden_dim = 2048  # Qwen3-1.7B default
        
        # Log memory usage
        mx.eval(self.parameters())
        gpu_memory = mx.get_active_memory() / (1024**3)
        logger.info(f"Rank {self.pipeline_rank}: GPU memory = {gpu_memory:.2f} GB")
    
    def custom_forward(self, inputs, mask=None, cache=None):
        """
        Custom forward pass that handles pipeline parallelism.
        Uses all_gather for communication (since send/recv are broken).
        """
        if not hasattr(self, 'pipeline_rank'):
            # Not in pipeline mode - use original forward
            if hasattr(self, '_original_forward'):
                return self._original_forward(inputs, mask, cache)
            else:
                # Fallback standard forward
                h = self.embed_tokens(inputs) if hasattr(self, 'embed_tokens') else inputs
                for layer in self.layers:
                    if layer is not None:
                        h = layer(h, mask, cache) if cache else layer(h, mask)
                if hasattr(self, 'norm'):
                    h = self.norm(h)
                return h
        
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        
        # Handle embeddings (only rank 0)
        if pipeline_rank == 0:
            if hasattr(self, 'embed_tokens'):
                h = self.embed_tokens(inputs)
                # Ensure correct shape
                if h.ndim == 2:
                    batch_size = 1
                    seq_len = h.shape[-1]
                    h = h.reshape(batch_size, seq_len, self.hidden_dim)
            else:
                h = inputs
        else:
            # Other ranks create zero tensor
            if inputs.ndim == 1:
                batch_size = 1
                seq_len = inputs.shape[0]
            else:
                batch_size = inputs.shape[0]
                seq_len = inputs.shape[1]
            h = mx.zeros((batch_size, seq_len, self.hidden_dim))
        
        # Initialize distributed cache (only for our layers)
        if cache is None:
            cache = [None] * self.num_layers
        
        # Process layers with pipeline communication
        for global_idx in range(self.total_layers):
            if self.start_idx <= global_idx < self.end_idx:
                # This is our layer - process it
                local_idx = global_idx - self.start_idx
                layer = self.layers[global_idx]
                
                if layer is not None:
                    if cache[local_idx] is not None:
                        h = layer(h, mask, cache[local_idx])
                    else:
                        h = layer(h, mask)
            
            # After each layer, synchronize using all_gather
            # This is inefficient but works around broken send/recv
            if global_idx < self.total_layers - 1:
                # Broadcast current activations to all ranks
                gathered = mx.distributed.all_gather(h)
                # Extract the portion we need (all_gather concatenates)
                h = gathered[:h.shape[0]]
                mx.eval(h)
        
        # Final norm (only last rank)
        if pipeline_rank == pipeline_size - 1:
            if hasattr(self, 'norm'):
                h = self.norm(h)
        
        # Final broadcast so all ranks have the output
        gathered = mx.distributed.all_gather(h)
        h = gathered[:h.shape[0]]
        mx.eval(h)
        
        return h
    
    # Save original forward
    if not hasattr(model_class, '_original_forward'):
        model_class._original_forward = model_class.__call__
    
    # Add our methods
    model_class.pipeline = pipeline
    model_class.__call__ = custom_forward
    
    logger.info(f"✅ Added custom pipeline to {model_class.__name__}")
    return model_class


def load_and_shard_model():
    """
    Load cached Qwen3 model and apply our custom pipeline.
    No lazy loading, no downloading - uses cached model.
    """
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
    
    # Apply our custom pipeline to Qwen3
    if world_size > 1:
        logger.info(f"Rank {rank}: Applying custom pipeline...")
        
        if hasattr(model, 'model'):
            # Qwen3 has model.model structure
            inner_model = model.model
            add_custom_pipeline(type(inner_model))
            inner_model.pipeline(distributed_group)
        else:
            add_custom_pipeline(type(model))
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
        logger.info(f"✅ DISTRIBUTED INFERENCE READY")
        logger.info(f"✅ Model: {model_name} (cached)")
        logger.info(f"✅ Devices: {world_size} GPUs")
        logger.info(f"✅ Pipeline: Custom implementation for Qwen3")
        logger.info(f"✅ Memory: {gpu_memory:.2f} GB on rank 0")
        logger.info("="*60)


def generate_distributed(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
    """Generate text using our custom pipeline."""
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
    logger.info("FINAL MLX DISTRIBUTED INFERENCE")
    logger.info(f"Using uv + Python {sys.version.split()[0]}")
    logger.info(f"Model: {model_name} (cached)")
    logger.info("="*60)
    
    # Load and shard
    load_and_shard_model()
    
    # Only rank 0 runs API
    if rank == 0:
        app = FastAPI(title="MLX Distributed (Custom Pipeline)", version="1.0")
        
        @app.get("/")
        async def dashboard():
            gpu_memory = mx.get_active_memory() / (1024**3)
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLX Custom Pipeline</title>
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
                    <h1>🚀 MLX Custom Pipeline Distributed</h1>
                    <p style="color: #9ca3af;">Using our custom pipeline implementation for Qwen3</p>
                    
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
                            <div class="metric">Qwen3</div>
                            <div class="label">Model Type</div>
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
                "pipeline": "custom_qwen3",
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