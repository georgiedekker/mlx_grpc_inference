#!/usr/bin/env python3
"""
Single-device API server for MLX inference (no distributed processing).

This bypasses the distributed worker issues and provides correct outputs
while we fix the worker model weight problems.
"""

import asyncio
import time
import uuid
import logging
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None

# API Models (OpenAI-compatible)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mlx-community/Qwen3-1.7B-8bit"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int  
    total_tokens: int
    prompt_tokens_per_second: Optional[float] = None
    generation_tokens_per_second: Optional[float] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global model, tokenizer
    
    logger.info("🚀 Starting Single-Device MLX Inference API...")
    
    # Load model and tokenizer
    logger.info("📦 Loading model: mlx-community/Qwen3-1.7B-8bit")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    logger.info(f"✅ Model loaded: {len(model.layers)} layers")
    
    logger.info("✅ Single-device API server ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down API server...")

app = FastAPI(
    lifespan=lifespan, 
    title="MLX Single-Device Inference API",
    description="OpenAI-compatible API with single-device processing (no distribution)", 
    version="1.0.0"
)

def format_messages(messages: List[ChatMessage]) -> str:
    """Format messages into a prompt string."""
    prompt = ""
    for msg in messages:
        prompt += f"{msg.role}: {msg.content}\\n"
    prompt += "assistant: "
    return prompt

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {"message": "MLX Single-Device Inference API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": "single_device",
        "model_loaded": model is not None,
        "distributed": False
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using single-device inference."""
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        overall_start_time = time.time()
        
        # Format and tokenize
        prompt = format_messages(request.messages)
        tokenize_start = time.time()
        input_ids = mx.array(tokenizer.encode(prompt))
        prompt_tokens = len(input_ids)
        tokenize_time = time.time() - tokenize_start
        
        logger.info(f"🚀 Single-device inference for {prompt_tokens} prompt tokens")
        
        # Ensure proper shape
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        
        # Prompt processing
        prompt_start_time = time.time()
        
        # Process through all layers locally
        hidden_states = model.model.embed_tokens(input_ids)
        
        for layer in model.model.layers:
            layer_output = layer(hidden_states)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
        
        # Final norm and logits
        hidden_states = model.model.norm(hidden_states)
        logits = model.model.embed_tokens.as_linear(hidden_states)
        
        prompt_end_time = time.time()
        prompt_processing_time = prompt_end_time - prompt_start_time
        prompt_tokens_per_second = prompt_tokens / prompt_processing_time if prompt_processing_time > 0 else 0
        
        # Generation
        generation_start_time = time.time()
        sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
        generated_ids = []
        
        current_ids = input_ids
        max_new_tokens = min(request.max_tokens or 50, 100)
        
        for token_idx in range(max_new_tokens):
            # Sample next token
            next_token = sampler(logits[:, -1:, :])
            token_id = next_token.item()
            generated_ids.append(token_id)
            
            # Check for stop conditions
            if token_id == tokenizer.eos_token_id:
                break
            if request.stop and tokenizer.decode([token_id]) in request.stop:
                break
            
            # Append new token and continue
            new_token_tensor = mx.array([[token_id]])
            current_ids = mx.concatenate([current_ids, new_token_tensor], axis=1)
            
            # Forward pass for next token
            hidden_states = model.model.embed_tokens(current_ids)
            for layer in model.model.layers:
                layer_output = layer(hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
            
            hidden_states = model.model.norm(hidden_states)
            logits = model.model.embed_tokens.as_linear(hidden_states)
        
        generation_end_time = time.time()
        generation_time = generation_end_time - generation_start_time
        completion_tokens = len(generated_ids)
        generation_tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
        
        # Decode response
        generated_text = tokenizer.decode(generated_ids) if generated_ids else ""
        
        # Log performance
        total_time = time.time() - overall_start_time
        logger.info(f"✅ Single-device Performance:")
        logger.info(f"   Prompt: {prompt_tokens} tokens @ {prompt_tokens_per_second:.1f} tok/s")
        logger.info(f"   Generation: {completion_tokens} tokens @ {generation_tokens_per_second:.1f} tok/s")
        logger.info(f"   Total time: {total_time:.2f}s")
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_per_second=round(prompt_tokens_per_second, 1),
                generation_tokens_per_second=round(generation_tokens_per_second, 1)
            )
        )
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Error in single-device completion: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "mlx-community/Qwen3-1.7B-8bit",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-community",
                "mode": "single_device"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)