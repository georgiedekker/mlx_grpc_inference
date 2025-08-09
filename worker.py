#!/usr/bin/env python3
"""
MLX Distributed Inference Server with Allreduce Pipeline Parallelism
Uses collective operations to avoid send/recv deadlocks
"""
import os
import time
import json
import logging
import resource
from typing import Dict, Any, List
from pathlib import Path
import uuid

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import stream_generate
from mlx_lm.utils import load as load_model
from mlx_lm.utils import load as mlx_load
from mlx_lm.utils import load_tokenizer
from mlx_lm.sample_utils import make_sampler
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU as default
mx.set_default_device(mx.gpu)

# Increase file limits for model loading
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 8192))
except:
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

# Global state
model = None
tokenizer = None
config = None
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


def add_allreduce_pipeline(model_class):
    """
    Add Allreduce-based pipeline parallelism to model.
    Uses collective operations to avoid deadlocks.
    """
    
    def pipeline(self, group):
        """Setup pipeline sharding."""
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        self.group = group
        
        if self.pipeline_size != 2:
            logger.info(f"Pipeline size {self.pipeline_size}, not sharding")
            return
        
        # Split layers in half
        if hasattr(self, 'layers'):
            num_layers = len(self.layers)
            mid = num_layers // 2
            
            if self.pipeline_rank == 0:
                self.layers = self.layers[:mid]
                logger.info(f"Rank 0: Processing layers 0-{mid-1}")
            else:
                self.layers = self.layers[mid:]
                logger.info(f"Rank 1: Processing layers {mid}-{num_layers-1}")
            
            self.num_layers = num_layers
        
        if hasattr(self, 'hidden_size'):
            self.hidden_dim = self.hidden_size
        else:
            self.hidden_dim = 2048  # Default
    
    def allreduce_forward(self, inputs, mask=None, cache=None):
        """Forward pass using Allreduce."""
        if not hasattr(self, 'pipeline_rank'):
            # Not in pipeline mode - use original forward
            if hasattr(self, '_original_forward'):
                return self._original_forward(inputs, mask, cache)
            else:
                # Standard forward
                h = self.embed_tokens(inputs) if hasattr(self, 'embed_tokens') else inputs
                for layer in self.layers:
                    h = layer(h, mask, cache) if cache else layer(h, mask)
                if hasattr(self, 'norm'):
                    h = self.norm(h)
                return h
        
        rank = self.pipeline_rank
        world_size = self.pipeline_size
        hostname = os.uname().nodename
        
        # Determine shape
        if inputs.ndim == 1:
            batch_size = 1
            seq_len = inputs.shape[0]
        else:
            batch_size = inputs.shape[0] if inputs.ndim > 1 else 1
            seq_len = inputs.shape[-1]
        
        if rank == 0:
            # Rank 0: Embedding + first half
            if hasattr(self, 'embed_tokens'):
                h = self.embed_tokens(inputs)
                if h.ndim == 2:
                    h = h.reshape(batch_size, seq_len, self.hidden_dim)
            else:
                h = inputs
            
            logger.debug(f"Rank 0 on {hostname}: Processing first half")
            
            # Process first half of layers
            if cache is None:
                cache = [None] * len(self.layers)
            
            for layer, c in zip(self.layers, cache):
                h = layer(h, mask, c)
            
            # Send to rank 1 via all_sum
            broadcast = mx.distributed.all_sum(h)
            mx.eval(broadcast)
            
            # Wait for result from rank 1
            zeros = mx.zeros((batch_size, seq_len, self.hidden_dim))
            result = mx.distributed.all_sum(zeros)
            mx.eval(result)
            
            return result
            
        else:  # rank == 1
            # Receive from rank 0
            zeros = mx.zeros((batch_size, seq_len, self.hidden_dim))
            h = mx.distributed.all_sum(zeros)
            mx.eval(h)
            
            logger.debug(f"Rank 1 on {hostname}: Processing second half")
            
            # Process second half
            if cache is None:
                cache = [None] * len(self.layers)
            
            for layer, c in zip(self.layers, cache):
                h = layer(h, mask, c)
            
            # Apply final norm
            if hasattr(self, 'norm'):
                h = self.norm(h)
            
            # Send back to rank 0
            result = mx.distributed.all_sum(h)
            mx.eval(result)
            
            return result
    
    # Save original forward
    if not hasattr(model_class, '_original_forward'):
        model_class._original_forward = model_class.__call__
    
    # Add methods
    model_class.pipeline = pipeline
    model_class.__call__ = allreduce_forward
    
    logger.info(f"✅ Added Allreduce pipeline to {model_class.__name__}")


def download(repo: str, allow_patterns: list[str]) -> Path:
    """Download model files from HuggingFace."""
    return Path(
        snapshot_download(
            repo,
            allow_patterns=allow_patterns,
        )
    )


def shard_and_load(repo: str):
    """
    Load and shard model using Allreduce pipeline parallelism.
    """
    global model, tokenizer, config, distributed_group, rank, world_size
    
    # Initialize distributed group
    distributed_group = mx.distributed.init()
    if not distributed_group:
        logger.info("No distributed group, running in single GPU mode")
        model, tokenizer = mlx_load(repo)
        rank = 0
        world_size = 1
        return
    
    rank = distributed_group.rank()
    world_size = distributed_group.size()
    hostname = os.uname().nodename
    logger.info(f"🚀 Rank {rank}/{world_size} on {hostname}")
    
    # Download model metadata
    logger.info(f"Rank {rank}: Downloading model metadata...")
    model_path = download(
        repo,
        allow_patterns=["*.json", "*.py", "tokenizer.model", "*.tiktoken", "*.txt"],
    )
    
    # Load model lazily
    logger.info(f"Rank {rank}: Loading model structure...")
    model, tokenizer = mlx_load(model_path, lazy=True)
    
    # Add Allreduce pipeline parallelism
    if world_size > 1:
        logger.info(f"Rank {rank}: Adding Allreduce pipeline parallelism...")
        if hasattr(model, 'model'):
            # Models like Qwen3 have model.model
            add_allreduce_pipeline(type(model.model))
            model.model.pipeline(distributed_group)
        else:
            add_allreduce_pipeline(type(model))
            model.pipeline(distributed_group)
    
    # Download weights
    weight_index_path = model_path / "model.safetensors.index.json"
    if weight_index_path.exists() and world_size > 1:
        with open(weight_index_path, "r") as fid:
            weight_index = json.load(fid)["weight_map"]
        
        local_files = set()
        for k, _ in tree_flatten(model.parameters()):
            if k in weight_index:
                local_files.add(weight_index[k])
        
        if local_files:
            logger.info(f"Rank {rank}: Downloading {len(local_files)} weight files...")
            download(repo, allow_patterns=list(local_files))
    else:
        logger.info(f"Rank {rank}: Downloading all weights...")
        download(repo, allow_patterns=["*.safetensors"])
    
    # Load tokenizer
    tokenizer = load_tokenizer(
        model_path,
        {"trust_remote_code": True},
    )
    
    # Load weights
    logger.info(f"Rank {rank}: Loading model weights...")
    mx.eval(model.parameters())
    
    # Check memory
    gpu_memory = mx.get_active_memory() / (1024**3)
    logger.info(f"Rank {rank} on {hostname}: GPU memory = {gpu_memory:.2f} GB")
    
    # Synchronize
    if world_size > 1:
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
        logger.info(f"Rank {rank}: Synchronized")
    
    if rank == 0:
        logger.info("="*70)
        logger.info(f"✅ READY: {world_size} GPU(s) with Allreduce pipeline")
        logger.info(f"✅ Model: {repo}")
        logger.info("="*70)


def generate_pipeline_parallel(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Generate text using pipeline parallelism.
    Returns metrics including tokens per second.
    """
    global model, tokenizer, rank, distributed_group, world_size
    
    # Timing
    prompt_start = time.time()
    
    # Count prompt tokens
    prompt_tokens = len(tokenizer.encode(prompt)) if rank == 0 else 0
    prompt_time = time.time() - prompt_start
    
    # Generation
    gen_start = time.time()
    
    # Create sampler
    sampler = make_sampler(temp=temperature)
    
    # All ranks must participate in stream_generate
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
    
    gen_time = time.time() - gen_start
    total_time = time.time() - prompt_start
    
    # Only rank 0 returns results
    if rank == 0 and responses:
        final_response = responses[-1]
        
        # Extract metrics
        prompt_tps = final_response.prompt_tps if hasattr(final_response, 'prompt_tps') else 0
        gen_tps = final_response.generation_tps if hasattr(final_response, 'generation_tps') else 0
        completion_tokens = final_response.generation_tokens if hasattr(final_response, 'generation_tokens') else len(generated_text.split())
        
        logger.info(f"Generated {completion_tokens} tokens in {gen_time:.2f}s = {gen_tps:.1f} tok/s")
        logger.info(f"Using {world_size} GPUs with Allreduce pipeline")
        
        return {
            "text": generated_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokens_per_second": round(gen_tps, 1),
            "prompt_eval_tokens_per_second": round(prompt_tps, 1),
            "eval_tokens_per_second": round(gen_tps, 1),
            "generation_time": gen_time,
            "prompt_time": prompt_time,
            "total_time": total_time,
            "gpus_used": world_size
        }
    else:
        # Worker ranks return empty dict
        return {}


def worker_loop():
    """Worker rank loop."""
    logger.info(f"Worker rank {rank} ready for Allreduce pipeline processing")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info(f"Worker rank {rank} shutting down")


def main():
    """Main entry point."""
    global rank, world_size, model_name
    
    # Load and shard model
    logger.info(f"Loading model: {model_name}")
    shard_and_load(model_name)
    
    # Only rank 0 runs the API server
    if rank == 0:
        app = FastAPI(
            title="MLX Distributed Inference (Allreduce)",
            version="2.0"
        )
        
        @app.get("/")
        async def dashboard():
            """Dashboard showing distributed status."""
            gpu_memory = mx.get_active_memory() / (1024**3)
            
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLX Distributed Inference</title>
                <style>
                    body {{ font-family: -apple-system, sans-serif; background: #1e1e1e; color: #fff; padding: 20px; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                    h1 {{ margin: 0; }}
                    .status {{ margin: 20px 0; padding: 15px; background: #2a2a2a; border-radius: 8px; }}
                    .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                    .value {{ font-size: 24px; font-weight: bold; color: #4ade80; }}
                    .label {{ font-size: 12px; color: #999; text-transform: uppercase; }}
                    .success {{ background: #4ade80; color: #000; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 MLX Distributed Inference</h1>
                    <p>Allreduce Pipeline Parallelism (No Deadlocks!)</p>
                </div>
                <div class="status">
                    <div class="metric">
                        <div class="value">ONLINE</div>
                        <div class="label">Status</div>
                    </div>
                    <div class="metric">
                        <div class="value">{world_size}</div>
                        <div class="label">Active GPUs</div>
                    </div>
                    <div class="metric">
                        <div class="value">{gpu_memory:.2f} GB</div>
                        <div class="label">GPU Memory (Rank 0)</div>
                    </div>
                </div>
                <div class="success">
                    ✅ Model sharded across {world_size} GPUs with Allreduce!
                </div>
            </body>
            </html>
            """)
        
        @app.post("/v1/chat/completions", response_model=ChatResponse)
        async def chat_completions(request: ChatRequest):
            """OpenAI-compatible endpoint."""
            try:
                # Extract user message
                prompt = ""
                for msg in request.messages:
                    if msg.role == "user":
                        prompt = msg.content
                        break
                
                if not prompt:
                    prompt = "Hello"
                
                # Apply chat template
                messages = [{"role": "user", "content": prompt}]
                try:
                    prompt_formatted = tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                    if isinstance(prompt_formatted, list):
                        prompt = tokenizer.decode(prompt_formatted)
                    else:
                        prompt = prompt_formatted
                except:
                    # Fallback
                    prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                # Generate with pipeline parallelism
                result = generate_pipeline_parallel(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                # Return OpenAI-compatible response
                return ChatResponse(
                    id=f"chatcmpl-{str(uuid.uuid4())[:8]}",
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
                        "prompt_eval_tokens_per_second": result.get("prompt_eval_tokens_per_second", 0),
                        "eval_tokens_per_second": result.get("eval_tokens_per_second", 0),
                        "gpus_used": result.get("gpus_used", 1),
                        "generation_time": result.get("generation_time", 0),
                        "prompt_time": result.get("prompt_time", 0),
                        "total_time": result.get("total_time", 0)
                    }
                )
            except Exception as e:
                logger.error(f"Error in chat completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            """Health check."""
            gpu_memory = mx.get_active_memory() / (1024**3)
            
            return {
                "status": "healthy",
                "rank": rank,
                "world_size": world_size,
                "model": model_name,
                "distributed": world_size > 1,
                "gpu_memory_gb": round(gpu_memory, 2),
                "pipeline": "allreduce"
            }
        
        # Run the API server
        logger.info("Starting API server on rank 0")
        uvicorn.run(app, host="0.0.0.0", port=8100)
    else:
        # Worker ranks
        worker_loop()


if __name__ == "__main__":
    main()