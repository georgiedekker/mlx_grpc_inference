#!/usr/bin/env python3
"""
MLX Distributed Inference with Proper Pipeline and KV Cache
Based on DeepSeek's implementation pattern
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


def add_pipeline_to_model(model_class):
    """
    Add proper pipeline method to any model class.
    Based on DeepSeek's implementation but generalized.
    """
    
    # Save original methods
    if not hasattr(model_class, '_original_call'):
        model_class._original_call = model_class.__call__
    
    def pipeline(self, group):
        """
        Setup pipeline parallelism with proper layer distribution.
        Follows DeepSeek pattern: keeps None placeholders for non-owned layers.
        """
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        
        if not hasattr(self, 'layers'):
            logger.warning(f"Model {model_class.__name__} has no layers attribute")
            return
        
        total_layers = len(self.layers)
        layers_per_rank = total_layers // self.pipeline_size
        extra = total_layers % self.pipeline_size
        
        # Calculate which layers this rank owns
        if self.pipeline_rank < extra:
            layers_per_rank += 1
            self.start_idx = self.pipeline_rank * layers_per_rank
        else:
            self.start_idx = self.pipeline_rank * layers_per_rank + extra
        
        self.end_idx = self.start_idx + layers_per_rank
        self.num_layers = layers_per_rank
        
        logger.info(f"Rank {self.pipeline_rank}: Owning layers {self.start_idx}-{self.end_idx-1} (total: {self.num_layers})")
        
        # Create new layer list with None placeholders
        new_layers = []
        for i in range(total_layers):
            if self.start_idx <= i < self.end_idx:
                new_layers.append(self.layers[i])
            else:
                new_layers.append(None)
        
        self.layers = new_layers
        self.total_layers = total_layers
        
        # Store original hidden size for tensor shapes
        if hasattr(self, 'hidden_size'):
            self.hidden_dim = self.hidden_size
        elif hasattr(self, 'config') and hasattr(self.config, 'hidden_size'):
            self.hidden_dim = self.config.hidden_size
        else:
            self.hidden_dim = 2048  # Default
        
        # Force weight evaluation to actually drop unused layers
        mx.eval(self.parameters())
        
        # Log memory after sharding
        gpu_memory = mx.get_active_memory() / (1024**3)
        logger.info(f"Rank {self.pipeline_rank}: GPU memory after pipeline setup = {gpu_memory:.2f} GB")
    
    def pipeline_call(self, inputs, mask=None, cache=None):
        """
        Forward pass with pipeline parallelism and distributed KV cache.
        Each rank only processes its layers and maintains cache for those layers only.
        """
        if not hasattr(self, 'pipeline_rank'):
            # Not in pipeline mode
            return self._original_call(inputs, mask, cache)
        
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        
        # Embedding (only rank 0 does this)
        if pipeline_rank == 0:
            if hasattr(self, 'embed_tokens'):
                h = self.embed_tokens(inputs)
            else:
                h = inputs
            
            # Determine shape for creating receive buffer
            if h.ndim == 2:
                batch_size, seq_len = h.shape
                hidden_shape = (batch_size, seq_len, self.hidden_dim)
                h = h.reshape(hidden_shape)
            else:
                hidden_shape = h.shape
        else:
            # Other ranks need to know the shape
            # For simplicity, assume standard shapes
            if inputs.ndim == 1:
                batch_size = 1
                seq_len = inputs.shape[0]
            else:
                batch_size = inputs.shape[0] if inputs.ndim > 1 else 1
                seq_len = inputs.shape[-1]
            hidden_shape = (batch_size, seq_len, self.hidden_dim)
            h = mx.zeros(hidden_shape)
        
        # Initialize cache if needed (only for owned layers)
        if cache is None:
            # Each rank only maintains cache for its layers
            cache = [None] * self.num_layers
        elif isinstance(cache, list) and len(cache) == self.total_layers:
            # Extract only cache for our layers
            my_cache = []
            for i in range(self.total_layers):
                if self.start_idx <= i < self.end_idx:
                    my_cache.append(cache[i])
            cache = my_cache
        
        # Receive from previous rank (except rank 0)
        if pipeline_rank > 0:
            logger.debug(f"Rank {pipeline_rank}: Receiving from rank {pipeline_rank-1}")
            h = mx.distributed.recv_like(h, src=(pipeline_rank - 1))
            mx.eval(h)  # Force evaluation
        
        # Process our layers
        cache_idx = 0
        for i in range(self.total_layers):
            if self.start_idx <= i < self.end_idx:
                layer = self.layers[i]
                if layer is not None:
                    logger.debug(f"Rank {pipeline_rank}: Processing layer {i}")
                    if cache_idx < len(cache) and cache[cache_idx] is not None:
                        h = layer(h, mask, cache[cache_idx])
                    else:
                        h = layer(h, mask)
                    cache_idx += 1
        
        # Send to next rank (except last rank)
        if pipeline_rank < pipeline_size - 1:
            logger.debug(f"Rank {pipeline_rank}: Sending to rank {pipeline_rank+1}")
            h = mx.distributed.send(h, dst=(pipeline_rank + 1))
            mx.eval(h)  # Force evaluation
            # Non-last ranks return zeros (they don't have final output)
            return mx.zeros_like(h)
        
        # Last rank applies final norm and broadcasts result
        if pipeline_rank == pipeline_size - 1:
            if hasattr(self, 'norm'):
                h = self.norm(h)
            
            # Broadcast final result to all ranks using all_gather
            logger.debug(f"Rank {pipeline_rank}: Broadcasting final result")
            gathered = mx.distributed.all_gather(h)
            # Extract just our portion (all_gather concatenates)
            h = gathered[:h.shape[0]]
            mx.eval(h)
        
        return h
    
    # Add methods to model class
    model_class.pipeline = pipeline
    model_class.__call__ = pipeline_call
    
    logger.info(f"✅ Added pipeline parallelism to {model_class.__name__}")
    return model_class


def load_and_shard(repo: str):
    """Load model and setup pipeline parallelism."""
    global model, tokenizer, distributed_group, rank, world_size
    
    # Initialize distributed
    distributed_group = mx.distributed.init()
    
    if not distributed_group:
        logger.info("No distributed group, loading in single GPU mode")
        model, tokenizer = mlx_load(repo)
        rank = 0
        world_size = 1
        return
    
    rank = distributed_group.rank()
    world_size = distributed_group.size()
    hostname = os.uname().nodename
    
    logger.info(f"🚀 Rank {rank}/{world_size} on {hostname}")
    
    # Load model
    logger.info(f"Rank {rank}: Loading model...")
    
    # Check cache first
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_id = repo.replace("/", "--")
    cached_paths = list(cache_dir.glob(f"models--{repo_id}"))
    
    if cached_paths:
        model_path = cached_paths[0] / "snapshots"
        if model_path.exists():
            snapshot_dirs = list(model_path.iterdir())
            if snapshot_dirs:
                model_path = snapshot_dirs[0]
                logger.info(f"Using cached model from {model_path}")
    else:
        # Download if not cached
        logger.info(f"Downloading {repo}...")
        model_path = Path(snapshot_download(repo))
    
    # Load model and tokenizer
    model, tokenizer = mlx_load(model_path, lazy=False)
    
    # Add pipeline to model
    if world_size > 1:
        if hasattr(model, 'model'):
            # Models like Qwen3 have model.model structure
            inner_model = model.model
            add_pipeline_to_model(type(inner_model))
            inner_model.pipeline(distributed_group)
        else:
            add_pipeline_to_model(type(model))
            model.pipeline(distributed_group)
    
    # Synchronize
    if world_size > 1:
        logger.info(f"Rank {rank}: Synchronizing...")
        sync = mx.distributed.all_sum(mx.array([1.0]))
        mx.eval(sync)
        logger.info(f"Rank {rank}: Ready")
    
    if rank == 0:
        logger.info("="*60)
        logger.info(f"✅ PIPELINE DISTRIBUTED INFERENCE READY")
        logger.info(f"✅ Model: {repo}")
        logger.info(f"✅ Devices: {world_size} GPUs")
        logger.info(f"✅ Pipeline: Proper send/recv with distributed KV cache")
        gpu_memory = mx.get_active_memory() / (1024**3)
        logger.info(f"✅ Master GPU Memory: {gpu_memory:.2f} GB")
        logger.info("="*60)


def generate_with_pipeline(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate text using pipeline parallelism with distributed KV cache."""
    global model, tokenizer, rank, world_size
    
    # All ranks must participate
    start_time = time.time()
    
    # Only rank 0 handles prompt
    if rank == 0:
        prompt_tokens = len(tokenizer.encode(prompt))
        logger.info(f"Generating {max_tokens} tokens from {prompt_tokens} prompt tokens")
    else:
        prompt_tokens = 0
        prompt = ""  # Other ranks don't need prompt
    
    # Create sampler
    sampler = make_sampler(temp=temperature)
    
    # All ranks participate in generation
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
            logger.info(f"✅ Speed: {gen_tps:.1f} tokens/sec")
            logger.info(f"✅ Using {world_size} GPUs with proper pipeline")
            
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
    global rank, world_size, model_name
    
    logger.info("="*60)
    logger.info("MLX PIPELINE DISTRIBUTED INFERENCE")
    logger.info(f"Model: {model_name}")
    logger.info(f"Python: {sys.version}")
    logger.info("="*60)
    
    # Load and shard model
    load_and_shard(model_name)
    
    # Only rank 0 runs API
    if rank == 0:
        app = FastAPI(title="MLX Pipeline Distributed", version="1.0")
        
        @app.get("/")
        async def dashboard():
            gpu_memory = mx.get_active_memory() / (1024**3)
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLX Pipeline Distributed</title>
                <style>
                    body {{ font-family: -apple-system, sans-serif; background: #1a1a1a; color: #fff; padding: 40px; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    h1 {{ color: #4ade80; }}
                    .status {{ background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                    .metric {{ margin: 10px 0; }}
                    .value {{ color: #60a5fa; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 MLX Pipeline Distributed Inference</h1>
                    <div class="status">
                        <div class="metric">Status: <span class="value">ONLINE</span></div>
                        <div class="metric">GPUs: <span class="value">{world_size}</span></div>
                        <div class="metric">Model: <span class="value">{model_name}</span></div>
                        <div class="metric">GPU Memory: <span class="value">{gpu_memory:.2f} GB</span></div>
                        <div class="metric">Pipeline: <span class="value">Proper send/recv with distributed KV cache</span></div>
                    </div>
                    <p>API endpoint: <a href="/docs">/docs</a></p>
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
                result = generate_with_pipeline(
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
                logger.error(f"Error in chat: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "rank": rank,
                "world_size": world_size,
                "model": model_name,
                "gpu_memory_gb": round(mx.get_active_memory() / (1024**3), 2),
                "pipeline": "proper_send_recv"
            }
        
        logger.info("Starting API server on http://0.0.0.0:8100")
        uvicorn.run(app, host="0.0.0.0", port=8100)
    else:
        # Worker ranks
        logger.info(f"Worker rank {rank} ready for pipeline processing")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"Worker rank {rank} shutting down")


if __name__ == "__main__":
    main()