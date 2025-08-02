#!/usr/bin/env python3
"""
Simple MLX Inference Server - Single file working solution
Uses UV for all package management
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

# FastAPI for the server
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# MLX imports
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models to try in order of preference
MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit", 
    "mlx-community/Qwen2.5-3B-Instruct-4bit"
]

# Global model storage
model = None
tokenizer = None
model_name = None

# API Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]

# Create FastAPI app
app = FastAPI(title="Simple MLX Server")

def load_model(model_path: str = None):
    """Load the MLX model"""
    global model, tokenizer, model_name
    
    if model_path:
        models_to_try = [model_path]
    else:
        models_to_try = MODELS
    
    for model_path in models_to_try:
        try:
            logger.info(f"Loading model: {model_path}")
            model, tokenizer = load(model_path)
            model_name = model_path
            logger.info(f"✅ Model loaded successfully: {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load {model_path}: {e}")
            continue
    
    logger.error("❌ Failed to load any model")
    return False

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    if not load_model():
        logger.error("Failed to load model on startup")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple MLX Server",
        "model": model_name,
        "status": "ready" if model is not None else "no model loaded"
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    if model_name:
        return {
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx"
            }]
        }
    return {"data": []}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Handle chat completions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Build prompt from messages
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant: "
    
    # Generate response
    if request.stream:
        return StreamingResponse(
            stream_response(prompt, request),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            temp=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=model_name,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }]
        )

async def stream_response(prompt: str, request: ChatRequest):
    """Stream response generator"""
    try:
        # Start with opening
        chunk_id = f"chatcmpl-{int(time.time())}"
        
        # Opening chunk
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
        
        # Generate tokens
        for token in generate(
            model,
            tokenizer,
            prompt=prompt,
            temp=request.temperature,
            max_tokens=request.max_tokens,
            stream=True
        ):
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
        
        # Final chunk
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model": model_name,
        "timestamp": time.time()
    }

@app.get("/device-info")
async def device_info():
    """Get device information"""
    return {
        "device": str(mx.default_device()),
        "metal_available": mx.metal.is_available() if hasattr(mx, 'metal') else False,
        "model_loaded": model is not None,
        "model_name": model_name
    }

if __name__ == "__main__":
    import sys
    
    # Get port from command line or default to 8100
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
    
    logger.info(f"Starting Simple MLX Server on port {port}")
    logger.info(f"Device: {mx.default_device()}")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)