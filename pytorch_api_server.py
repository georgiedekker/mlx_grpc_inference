#!/usr/bin/env python3
"""
PyTorch Distributed API Server
Compatible with OpenAI API format, similar to your existing MLX server
"""
import os
import sys
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket
import datetime
import logging
import asyncio
from typing import Optional, List, Dict, Any
import time
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Import the distributed server components
from pytorch_distributed_server import DistributedInferenceEngine, setup_distributed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - Rank %(rank)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class RankLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {**kwargs, 'extra': {'rank': self.extra.get('rank', '?')}}

logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': '?'})

# API Models (same as your MLX server)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "microsoft/phi-2"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False

# Global engine instance
engine = None
rank = None
world_size = None

class APIInferenceEngine(DistributedInferenceEngine):
    """Extended inference engine with API-friendly methods"""
    
    async def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text from prompt string"""
        if self.rank != 0:
            # Only rank 0 handles API requests
            return ""
        
        # Tokenize input
        input_ids = self.model_shard.tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate
        start_time = time.time()
        output_ids = await asyncio.to_thread(
            self.generate, 
            input_ids, 
            max_tokens, 
            temperature
        )
        
        # Decode
        output_text = self.model_shard.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove prompt from output
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()
        
        generation_time = time.time() - start_time
        tokens_generated = len(output_ids[0]) - len(input_ids[0])
        
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_generated/generation_time:.1f} tok/s)")
        
        return output_text

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global engine, rank, world_size, logger
    
    logger.info("Starting PyTorch Distributed API Server")
    
    # Setup distributed
    rank, world_size = setup_distributed()
    
    # Update logger
    logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': rank})
    
    # Model configuration
    model_name = os.environ.get('MODEL_NAME', 'microsoft/phi-2')
    
    # Initialize inference engine
    logger.info(f"Creating inference engine for {model_name}")
    engine = APIInferenceEngine(model_name, rank, world_size)
    
    if rank == 0:
        logger.info("Rank 0: API server ready")
    else:
        # Start worker loop in background
        async def worker_task():
            await asyncio.to_thread(lambda: engine.generate(None, None, None))
        
        asyncio.create_task(worker_task())
        logger.info(f"Rank {rank}: Worker started")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()

# Create FastAPI app
app = FastAPI(
    title="PyTorch Distributed Inference",
    lifespan=lifespan,
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if rank != 0:
        raise HTTPException(status_code=503, detail="Only rank 0 serves API requests")
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Format prompt from messages
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant: "
    
    logger.info(f"Chat request: model={request.model}, max_tokens={request.max_tokens}")
    
    try:
        # Generate response
        response_text = await engine.generate_text(
            prompt,
            request.max_tokens,
            request.temperature
        )
        
        # Calculate token counts
        prompt_tokens = len(engine.model_shard.tokenizer.encode(prompt))
        completion_tokens = len(engine.model_shard.tokenizer.encode(response_text))
        
        # Build response
        response = {
            "id": f"chatcmpl-pytorch-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rank": rank,
        "world_size": world_size,
        "backend": "pytorch-gloo",
        "hostname": socket.gethostname(),
        "model_loaded": engine is not None and engine.model_shard is not None,
        "device": str(engine.model_shard.device) if engine else None
    }

@app.get("/models")
async def list_models():
    """List available models"""
    model_name = os.environ.get('MODEL_NAME', 'microsoft/phi-2')
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "owned_by": "system",
            "permission": []
        }]
    }

@app.get("/debug/distributed")
async def debug_distributed():
    """Debug distributed setup"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "distributed_initialized": dist.is_initialized() if world_size > 1 else False,
        "backend": dist.get_backend() if world_size > 1 and dist.is_initialized() else "none",
        "model_info": {
            "name": engine.model_shard.model_name,
            "device": str(engine.model_shard.device),
            "start_layer": engine.model_shard.start_layer,
            "end_layer": engine.model_shard.end_layer,
            "total_layers": len(engine.model_shard.assigned_layers)
        } if engine else None
    }

def main():
    """Main entry point"""
    # Only rank 0 runs the API server
    if rank == 0:
        logger.info("Starting API server on port 8100")
        uvicorn.run(app, host="0.0.0.0", port=8100)
    else:
        # Workers just keep the app context alive
        logger.info(f"Rank {rank}: Worker mode - no API server")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"Rank {rank}: Shutting down")

if __name__ == "__main__":
    # For workers that are started directly
    if int(os.environ.get('RANK', 0)) > 0:
        rank, world_size = setup_distributed()
        logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': rank})
        model_name = os.environ.get('MODEL_NAME', 'microsoft/phi-2')
        engine = APIInferenceEngine(model_name, rank, world_size)
        logger.info(f"Rank {rank}: Starting worker")
        engine.generate(None, None, None)  # Enter worker loop
    else:
        # Rank 0 starts through FastAPI
        main()