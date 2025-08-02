#!/usr/bin/env python3
"""
Simple MLX Server for master.local - standalone version
"""

import asyncio
import json
import logging
import time
import socket
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    logger.warning("MLX not available - running in fallback mode")
    MLX_AVAILABLE = False
    mx = None
    nn = None

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.error("FastAPI not available")
    FASTAPI_AVAILABLE = False

if not FASTAPI_AVAILABLE:
    print("Please install FastAPI: pip install fastapi uvicorn pydantic")
    exit(1)

# Models to try
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
app = FastAPI(title="Master MLX Server")

def load_model(model_path: str = None):
    """Load the MLX model"""
    global model, tokenizer, model_name
    
    if not MLX_AVAILABLE:
        logger.warning("MLX not available - using mock responses")
        model_name = "mock-model"
        return True
    
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
    """Initialize on startup"""
    hostname = socket.gethostname()
    logger.info(f"🚀 Starting MLX Server on {hostname}")
    logger.info(f"   Device: {'master.local' if 'master' in hostname else hostname}")
    logger.info(f"   MLX Available: {MLX_AVAILABLE}")
    
    if MLX_AVAILABLE:
        logger.info(f"   MLX Device: {mx.default_device()}")
        
    if not load_model():
        logger.warning("Failed to load model on startup - will use fallback")

@app.get("/")
async def root():
    """Root endpoint"""
    hostname = socket.gethostname()
    return {
        "message": "Master MLX Server",
        "device": "master.local",
        "hostname": hostname,
        "model": model_name,
        "mlx_available": MLX_AVAILABLE,
        "status": "ready" if model is not None or not MLX_AVAILABLE else "no model loaded"
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
                "owned_by": "mlx-master"
            }]
        }
    return {"data": []}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Handle chat completions"""
    try:
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
        
        logger.info(f"Processing chat completion: {request.model}, {len(request.messages)} messages")
        
        if not MLX_AVAILABLE or model is None:
            # Fallback response
            response = f"Hello! I'm the master.local server. I received your message: '{request.messages[-1].content}'. MLX is {'not available' if not MLX_AVAILABLE else 'available but no model loaded'}."
        else:
            # Use MLX generation
            if request.stream:
                return StreamingResponse(
                    stream_response(prompt, request),
                    media_type="text/event-stream"
                )
            else:
                response = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
        
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=model_name or "fallback",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }]
        )
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        logger.error(f"Request: {request}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def stream_response(prompt: str, request: ChatRequest):
    """Stream response generator"""
    try:
        chunk_id = f"chatcmpl-{int(time.time())}"
        
        # Opening chunk
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
        
        if MLX_AVAILABLE and model is not None:
            # Generate tokens
            for token in generate(
                model,
                tokenizer,
                prompt=prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            ):
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
        else:
            # Fallback streaming
            message = "Hello from master.local! "
            for char in message:
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'fallback', 'choices': [{'index': 0, 'delta': {'content': char}, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.1)
        
        # Final chunk
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name or 'fallback', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": "master.local",
        "model": model_name,
        "mlx_available": MLX_AVAILABLE,
        "timestamp": time.time()
    }

@app.get("/device-info")
async def device_info():
    """Get device information"""
    return {
        "device_id": "master",
        "hostname": "master.local",
        "port": 8102,
        "role": "worker",
        "layers": list(range(19, 28)),
        "mlx_device": str(mx.default_device()) if MLX_AVAILABLE else "N/A",
        "metal_available": mx.metal.is_available() if MLX_AVAILABLE and hasattr(mx, 'metal') else False,
        "model_loaded": model is not None,
        "model_name": model_name,
        "mlx_available": MLX_AVAILABLE
    }

if __name__ == "__main__":
    import sys
    
    # Use port 8102 for master
    port = 8102
    
    logger.info(f"Starting Master MLX Server on port {port}")
    if MLX_AVAILABLE:
        logger.info(f"MLX Device: {mx.default_device()}")
    else:
        logger.info("MLX not available - running in fallback mode")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)