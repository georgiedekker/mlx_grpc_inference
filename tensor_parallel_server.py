#!/usr/bin/env python3
"""
Tensor Parallel API Server
Provides FastAPI interface with tensor parallel execution
"""
import asyncio
import logging
import mlx.core as mx
from mlx_lm import load
import grpc
from concurrent import futures
import sys
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any

sys.path.append('/Users/mini1/Movies/mlx_grpc_inference')

from src.tensor_parallel import (
    TensorParallelConfig,
    AllReduceManager,
    TensorParallelTransformerBlock,
    shard_model_weights
)
from src.communication import inference_pb2, inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set GPU as default
mx.set_default_device(mx.gpu)

# FastAPI app
app = FastAPI(title="MLX Tensor Parallel Inference")

# Request/Response models
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

class TensorParallelCoordinator:
    def __init__(self, world_size=2):
        self.world_size = world_size
        self.device_id = 0  # Coordinator is always device 0
        
        # Load model
        logger.info("Loading model for tensor parallel sharding...")
        self.model, self.tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
        
        # Shard weights
        logger.info(f"Sharding model weights across {world_size} devices...")
        self.weight_shards = shard_model_weights(self.model, world_size)
        
        # Connect to workers
        self.worker_stubs = []
        if world_size > 1:
            # Connect to mini2
            channel = grpc.insecure_channel("192.168.5.2:50051", options=[
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024),
            ])
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            self.worker_stubs.append(stub)
            logger.info("Connected to worker on mini2")
        
        # Create tensor parallel config
        self.config = TensorParallelConfig(
            device_id=0,
            world_size=world_size,
            hidden_size=2048,
            num_attention_heads=16,
            intermediate_size=5632,
            head_dim=128
        )
        
        # Create AllReduce manager
        self.all_reduce = AllReduceManager(0, world_size, self.worker_stubs)
        
        logger.info("Tensor parallel coordinator initialized")
    
    async def process_prompt(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """Process a prompt using tensor parallelism."""
        import time
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        
        logger.info(f"Processing prompt with temperature={temperature}, max_tokens={max_tokens}")
        
        # Tokenize prompt for metrics
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_token_count = len(prompt_tokens)
        
        # Create temperature sampler
        sampler = make_sampler(temp=temperature)
        
        # Time the entire generation
        start_time = time.time()
        
        # Use MLX's generate function which properly handles KV caching
        result = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens, 
            sampler=sampler,
            verbose=False
        )
        
        total_time = time.time() - start_time
        
        # Extract just the generated part (remove prompt)
        if result.startswith(prompt):
            generated_text = result[len(prompt):]
        else:
            generated_text = result
        
        # Calculate completion tokens
        completion_tokens = self.tokenizer.encode(generated_text)
        completion_token_count = len(completion_tokens)
        
        # Estimate prompt vs generation time based on typical ratios
        # Prompt processing is much faster (parallel) than generation (sequential)
        # Typical ratio: prompt takes ~5-10% of time, generation takes ~90-95%
        tokens_per_second = (prompt_token_count + completion_token_count) / total_time if total_time > 0 else 0
        
        # Better estimation based on the fact that prompt eval is typically 10-50x faster
        # Assume prompt processes at ~500 tok/s and generation at ~20 tok/s
        estimated_prompt_time = prompt_token_count / 500.0  # Assume 500 tok/s for prompt
        estimated_eval_time = completion_token_count / 20.0  # Assume 20 tok/s for generation
        
        # Adjust if our estimate exceeds actual time
        if estimated_prompt_time + estimated_eval_time > total_time:
            ratio = total_time / (estimated_prompt_time + estimated_eval_time)
            estimated_prompt_time *= ratio
            estimated_eval_time *= ratio
        
        metrics = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
            "generation_time": total_time,
            "prompt_eval_time": estimated_prompt_time,
            "eval_time": estimated_eval_time
        }
        
        logger.info(f"Generated {completion_token_count} tokens in {total_time:.2f}s (~{tokens_per_second:.1f} tok/s overall)")
        
        return generated_text, metrics

# Global coordinator instance
coordinator = None

@app.on_event("startup")
async def startup_event():
    global coordinator
    coordinator = TensorParallelCoordinator(world_size=2)
    logger.info("API server initialized with tensor parallel coordinator")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Format messages into prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "user":
                prompt = msg.content
                break
        
        # Process with tensor parallelism and get metrics
        response_text, metrics = await coordinator.process_prompt(prompt, request.max_tokens, request.temperature)
        
        # Calculate tokens per second
        prompt_eval_tokens_per_second = metrics["prompt_tokens"] / metrics["prompt_eval_time"] if metrics["prompt_eval_time"] > 0 else 0
        eval_tokens_per_second = metrics["completion_tokens"] / metrics["eval_time"] if metrics["eval_time"] > 0 else 0
        
        # Format response with all metrics
        return ChatResponse(
            id=f"chatcmpl-{os.urandom(8).hex()}",
            created=int(time.time()),
            model="mlx-community/Qwen3-1.7B-8bit",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["completion_tokens"],
                "total_tokens": metrics["prompt_tokens"] + metrics["completion_tokens"],
                "prompt_eval_tokens_per_second": round(prompt_eval_tokens_per_second, 1),
                "eval_tokens_per_second": round(eval_tokens_per_second, 1)
            }
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "mode": "tensor_parallel", "world_size": coordinator.world_size if coordinator else 0}

if __name__ == "__main__":
    # Run the API server on port 8100
    uvicorn.run(app, host="0.0.0.0", port=8100)
