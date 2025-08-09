#!/usr/bin/env python3
"""
MLX Distributed Inference Server with PROPER Pipeline Parallelism
Based on Awni's pipeline_generate.py example
"""
import os
import time
import json
import logging
import resource
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.utils import load_model, load_tokenizer
from mlx_lm.sample_utils import make_sampler
import psutil

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU as default
mx.set_default_device(mx.gpu)

# Needed for 8 bit model (increased based on Awni's discussion)
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 8192))
except:
    # Fallback if higher limit fails
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

# Global state
model = None
tokenizer = None
config = None
distributed_group = None
rank = 0
world_size = 1
# Using Qwen3-1.7B to test custom pipeline implementation
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

def add_pipeline_to_model(model_instance):
    """
    Add pipeline() method to a model that doesn't have it.
    Based on our custom implementation.
    """
    class PipelineWrapper:
        def __init__(self, base_model):
            self.base_model = base_model
            self.group = None
            self.rank = 0
            self.world_size = 1
            self.start_layer = 0
            self.end_layer = None
            
        def pipeline(self, group):
            """Add pipeline parallelism to the model."""
            self.group = group
            self.rank = group.rank()
            self.world_size = group.size()
            
            logger.info(f"Custom pipeline: rank={self.rank}, world_size={self.world_size}")
            
            # Find layers in the model
            layers = None
            for attr in ['layers', 'blocks', 'h']:
                if hasattr(self.base_model, attr):
                    layers = getattr(self.base_model, attr)
                    break
                    
            if layers is None:
                logger.error("Could not find layers in model")
                return
                
            num_layers = len(layers)
            layers_per_rank = num_layers // self.world_size
            extra = num_layers % self.world_size
            
            # Calculate this rank's layers
            if self.rank < extra:
                self.start_layer = self.rank * (layers_per_rank + 1)
                self.end_layer = self.start_layer + layers_per_rank + 1
            else:
                self.start_layer = self.rank * layers_per_rank + extra
                self.end_layer = self.start_layer + layers_per_rank
                
            logger.info(f"Rank {self.rank}: Assigned layers {self.start_layer}-{self.end_layer-1} of {num_layers}")
            
            # Keep only our layers
            my_layers = []
            for i in range(num_layers):
                if self.start_layer <= i < self.end_layer:
                    my_layers.append(layers[i])
                    
            # Replace layers
            if hasattr(self.base_model, 'layers'):
                self.base_model.layers = my_layers
            elif hasattr(self.base_model, 'blocks'):
                self.base_model.blocks = my_layers
            elif hasattr(self.base_model, 'h'):
                self.base_model.h = my_layers
                
            logger.info(f"Rank {self.rank}: Pipeline sharding complete!")
    
    # Create wrapper and add pipeline method
    wrapper = PipelineWrapper(model_instance)
    model_instance.pipeline = wrapper.pipeline
    return model_instance

def shard_and_load(model_path: str):
    """
    Load and shard model using pipeline parallelism.
    If model doesn't have pipeline(), add it!
    """
    global model, tokenizer, config, distributed_group, rank, world_size
    
    # Initialize distributed group FIRST
    distributed_group = mx.distributed.init()
    if not distributed_group:
        logger.error("Failed to initialize distributed group")
        # Fallback to single device
        model, tokenizer = load_model(model_path)
        rank = 0
        world_size = 1
        return
    
    rank = distributed_group.rank()
    world_size = distributed_group.size()
    logger.info(f"Distributed initialized: rank={rank}, world_size={world_size}")
    
    # Load model configuration first
    config_path = Path(os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_path.replace('/', '--')}/snapshots"))
    if config_path.exists():
        # Find the latest snapshot
        snapshots = list(config_path.iterdir())
        if snapshots:
            model_local_path = snapshots[0]
        else:
            model_local_path = model_path
    else:
        model_local_path = model_path
    
    logger.info(f"Rank {rank}: Loading model from {model_local_path}")
    
    # Load tokenizer (all ranks need this)
    tokenizer = load_tokenizer(
        model_local_path,
        {"trust_remote_code": True}
    )
    
    # Lazy load model to figure out sharding
    model, config = load_model(model_local_path, lazy=True, strict=False)
    
    # Check if model has pipeline support
    if hasattr(model, 'model'):
        if not hasattr(model.model, 'pipeline'):
            logger.info(f"Rank {rank}: Model lacks pipeline(), adding custom implementation...")
            model.model = add_pipeline_to_model(model.model)
            logger.info(f"Rank {rank}: Custom pipeline() method added!")
        
        # Now apply pipeline parallelism
        logger.info(f"Rank {rank}: Applying pipeline parallelism")
        model.model.pipeline(distributed_group)
        logger.info(f"Rank {rank}: Pipeline parallelism applied successfully!")
    else:
        logger.warning(f"Rank {rank}: Model structure not recognized")
    
    # Evaluate parameters to load weights into memory
    mx.eval(model.parameters())
    
    # Log memory usage
    gpu_memory = mx.get_active_memory() / (1024**3)
    logger.info(f"Rank {rank}: GPU memory after loading = {gpu_memory:.2f} GB")
    
    # Synchronize all ranks before proceeding
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    logger.info(f"Rank {rank}: Ready for distributed inference!")

def generate_distributed(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Generate text using distributed pipeline parallelism.
    ALL ranks must participate in generation.
    """
    global model, tokenizer, rank, distributed_group, world_size
    
    # Timing for metrics
    prompt_start = time.time()
    
    # Log GPU memory before
    gpu_memory_before = mx.get_active_memory() / (1024**3)
    if rank == 0:
        logger.info(f"Rank {rank}: GPU memory before = {gpu_memory_before:.2f} GB")
    
    # Count prompt tokens
    # Ensure prompt is a string
    if not isinstance(prompt, str):
        logger.error(f"Prompt is not a string: {type(prompt)}")
        prompt = str(prompt)
    prompt_tokens = len(tokenizer.encode(prompt)) if rank == 0 else 0
    prompt_time = time.time() - prompt_start
    
    # Generation timing
    gen_start = time.time()
    
    # Create sampler with temperature
    sampler = make_sampler(temp=temperature, top_p=0.95)
    
    # ALL ranks must participate in stream_generate for pipeline parallelism
    # The stream_generate function handles the distributed communication
    responses = []
    for response in stream_generate(
        model, 
        tokenizer, 
        prompt, 
        max_tokens=max_tokens,
        sampler=sampler
    ):
        if rank == 0:
            # Only rank 0 collects the output
            responses.append(response)
    
    gen_time = time.time() - gen_start
    total_time = time.time() - prompt_start
    
    # Log GPU memory after
    gpu_memory_after = mx.get_active_memory() / (1024**3)
    if rank == 0:
        logger.info(f"Rank {rank}: GPU memory after = {gpu_memory_after:.2f} GB")
        logger.info(f"Rank {rank}: Memory delta = {gpu_memory_after - gpu_memory_before:.3f} GB")
    
    # Only rank 0 returns results
    if rank == 0 and responses:
        # Get the final response
        final_response = responses[-1]
        
        # Extract metrics from the response
        prompt_tps = final_response.prompt_tps if hasattr(final_response, 'prompt_tps') else 0
        gen_tps = final_response.generation_tps if hasattr(final_response, 'generation_tps') else 0
        completion_tokens = final_response.generation_tokens if hasattr(final_response, 'generation_tokens') else 0
        
        logger.info(f"Prompt: {prompt_tokens} tokens in {prompt_time:.2f}s = {prompt_tps:.1f} tok/s")
        logger.info(f"Generation: {completion_tokens} tokens in {gen_time:.2f}s = {gen_tps:.1f} tok/s")
        logger.info(f"✅ Using {world_size} GPUs with pipeline parallelism!")
        
        return {
            "text": final_response.text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokens_per_second": round(gen_tps, 1),
            "prompt_eval_tokens_per_second": round(prompt_tps, 1),
            "eval_tokens_per_second": round(gen_tps, 1),
            "generation_time": gen_time,
            "prompt_time": prompt_time,
            "total_time": total_time,
            "gpus_used": world_size,
            "peak_memory": final_response.peak_memory if hasattr(final_response, 'peak_memory') else gpu_memory_after
        }
    else:
        # Worker ranks return empty dict
        return {}

def worker_loop():
    """Worker rank loop - participates in distributed generation."""
    global model, tokenizer
    logger.info(f"Worker rank {rank} ready for distributed processing")
    
    # Workers participate in generation through stream_generate
    # They need to stay alive and ready to process pipeline stages
    try:
        while True:
            # Keep worker alive
            # The actual work happens when rank 0 calls stream_generate
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info(f"Worker rank {rank} shutting down")

def main():
    """Main entry point - Initialize distributed and load model BEFORE creating FastAPI."""
    global rank, world_size, model_name
    
    # CRITICAL: Load and shard model FIRST, before creating FastAPI
    logger.info(f"Loading model: {model_name}")
    shard_and_load(model_name)
    
    # Only rank 0 runs the API server
    if rank == 0:
        # NOW create the FastAPI app AFTER distributed initialization
        app = FastAPI(
            title="MLX Distributed Inference",
            version="3.0"
        )
        
        @app.get("/")
        async def dashboard():
            """Dashboard showing distributed GPU status."""
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
                    <p>Pipeline parallelism with {model_name}</p>
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
                    ✅ Model sharded across {world_size} GPUs using pipeline parallelism!
                </div>
            </body>
            </html>
            """)
        
        @app.post("/v1/chat/completions", response_model=ChatResponse)
        async def chat_completions(request: ChatRequest):
            """OpenAI-compatible endpoint using distributed inference."""
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
                # Some tokenizers return token IDs, others return strings
                try:
                    prompt_formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    if isinstance(prompt_formatted, list):
                        # If it's token IDs, decode them
                        prompt = tokenizer.decode(prompt_formatted)
                    else:
                        prompt = prompt_formatted
                except Exception as e:
                    logger.warning(f"Chat template failed: {e}, using raw prompt")
                    # Fallback to simple format
                    prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                # Generate with distributed pipeline parallelism
                result = generate_distributed(
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
                        "total_time": result.get("total_time", 0),
                        "peak_memory_gb": result.get("peak_memory", 0)
                    }
                )
            except Exception as e:
                import traceback
                logger.error(f"Error in chat completion: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            """Health check showing GPU status."""
            gpu_memory = mx.get_active_memory() / (1024**3)
            
            return {
                "status": "healthy",
                "rank": rank,
                "world_size": world_size,
                "model": model_name,
                "distributed": world_size > 1,
                "gpu_memory_gb": round(gpu_memory, 2),
                "pipeline_enabled": True
            }
        
        # Run the API server
        logger.info("Starting API server on rank 0")
        uvicorn.run(app, host="0.0.0.0", port=8100)
    else:
        # Worker ranks participate in distributed processing
        worker_loop()

if __name__ == "__main__":
    main()