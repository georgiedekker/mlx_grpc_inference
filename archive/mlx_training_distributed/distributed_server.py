#!/usr/bin/env python3
"""
Distributed MLX Inference Server - Multi-device support
Runs on port 8100 and coordinates with other devices
"""

import asyncio
import json
import logging
import time
import socket
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

# FastAPI for the server
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# MLX imports
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler

# Distributed inference imports
from distributed_inference import DistributedInferenceEngine, InferenceRequest
from grpc_tensor_service import start_grpc_server

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
@dataclass
class DeviceConfig:
    device_id: str
    hostname: str
    port: int
    role: str  # "coordinator" or "worker"
    user_home: str  # Home directory for this device
    layers: Optional[List[int]] = None
    status: str = "offline"

# Global configuration
DEVICES = [
    DeviceConfig("mini1", "mini1.local", 8100, "coordinator", "/Users/mini1", layers=list(range(0, 10))),
    DeviceConfig("mini2", "mini2.local", 8101, "worker", "/Users/mini2", layers=list(range(10, 19))),
    DeviceConfig("master", "master.local", 8102, "worker", "/Users/georgedekker", layers=list(range(19, 28)))
]

# Models to try
MODELS = [
    "mlx-community/Qwen3-1.7B-8bit",
    "mlx-community/GLM-4.5-4bit"
]

# Global state
model = None
tokenizer = None
model_name = None
device_registry = {}
local_device = None
distributed_engine = None

# API Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "mlx-community/Qwen3-1.7B-8bit"
    messages: List[ChatMessage] = [{"role": "user", "content": "Hi mom!"}]
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
app = FastAPI(title="Distributed MLX Server")

def get_local_device() -> DeviceConfig:
    """Determine which device this is based on hostname"""
    hostname = socket.gethostname().lower()
    
    for device in DEVICES:
        if device.device_id in hostname or hostname.startswith(device.device_id):
            return device
    
    # Default to first device if not found
    logger.warning(f"Unknown hostname {hostname}, defaulting to {DEVICES[0].device_id}")
    return DEVICES[0]

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

async def register_with_coordinator():
    """Register this device with the coordinator"""
    global local_device
    
    if local_device.role == "coordinator":
        # We are the coordinator
        device_registry[local_device.device_id] = {
            "config": local_device,
            "last_seen": time.time()
        }
        local_device.status = "online"
        return
    
    # Worker registering with coordinator
    coordinator = next(d for d in DEVICES if d.role == "coordinator")
    
    try:
        import requests
        response = requests.post(
            f"http://{coordinator.hostname}:{coordinator.port}/register",
            json={
                "device_id": local_device.device_id,
                "hostname": local_device.hostname,
                "port": local_device.port,
                "layers": local_device.layers
            },
            timeout=5
        )
        if response.status_code == 200:
            local_device.status = "online"
            logger.info(f"✅ Registered with coordinator at {coordinator.hostname}")
        else:
            logger.error(f"Failed to register: {response.text}")
    except Exception as e:
        logger.error(f"Failed to register with coordinator: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global local_device, distributed_engine
    
    # Determine which device we are
    local_device = get_local_device()
    logger.info(f"🚀 Starting as {local_device.device_id} ({local_device.role})")
    logger.info(f"   Hostname: {local_device.hostname}")
    logger.info(f"   Port: {local_device.port}")
    logger.info(f"   Layers: {local_device.layers}")
    
    # Load model (keeping for compatibility)
    if not load_model():
        logger.error("Failed to load model on startup")
    
    # Initialize distributed inference engine
    is_coordinator = (local_device.role == "coordinator")
    distributed_engine = DistributedInferenceEngine(local_device.device_id, is_coordinator)
    
    # Initialize the engine in background
    asyncio.create_task(initialize_distributed_engine())
    
    # Start gRPC server for tensor communication
    grpc_port = local_device.port + 1000  # 9100, 9101, 9102
    asyncio.create_task(start_grpc_tensor_server(grpc_port))
    
    # Register with coordinator (if we're a worker)
    await register_with_coordinator()
    
    # Start heartbeat if coordinator
    if local_device.role == "coordinator":
        asyncio.create_task(monitor_workers())

async def initialize_distributed_engine():
    """Initialize distributed inference engine"""
    global distributed_engine
    try:
        if distributed_engine:
            logger.info("Initializing distributed inference engine...")
            success = await distributed_engine.initialize(model_name or MODELS[0])
            if success:
                logger.info("✅ Distributed inference engine ready")
            else:
                logger.error("❌ Failed to initialize distributed inference engine")
    except Exception as e:
        logger.error(f"Error initializing distributed engine: {e}")

async def start_grpc_tensor_server(grpc_port: int):
    """Start gRPC tensor server for this device"""
    try:
        logger.info(f"Starting gRPC tensor server on port {grpc_port}")
        await start_grpc_server(
            local_device.device_id,
            local_device.hostname,
            grpc_port,
            local_device.layers or []
        )
    except Exception as e:
        logger.error(f"Failed to start gRPC tensor server: {e}")

async def monitor_workers():
    """Monitor worker health (coordinator only)"""
    while True:
        await asyncio.sleep(10)
        
        current_time = time.time()
        for device_id, info in device_registry.items():
            if current_time - info["last_seen"] > 30:
                info["config"].status = "offline"
                logger.warning(f"Device {device_id} is offline")

@app.post("/register")
async def register_device(request: Dict[str, Any]):
    """Register a worker device (coordinator only)"""
    if local_device.role != "coordinator":
        raise HTTPException(status_code=403, detail="Only coordinator can register devices")
    
    device_id = request.get("device_id")
    device_config = next((d for d in DEVICES if d.device_id == device_id), None)
    
    if not device_config:
        raise HTTPException(status_code=404, detail=f"Unknown device: {device_id}")
    
    device_registry[device_id] = {
        "config": device_config,
        "last_seen": time.time()
    }
    device_config.status = "online"
    
    logger.info(f"✅ Registered device: {device_id}")
    return {"status": "registered", "device_id": device_id}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Distributed MLX Server",
        "device": local_device.device_id,
        "role": local_device.role,
        "model": model_name,
        "status": local_device.status
    }

@app.get("/cluster-status")
async def cluster_status():
    """Get cluster status (coordinator only)"""
    if local_device.role != "coordinator":
        # Workers return their own status
        return {
            "device": local_device.device_id,
            "status": local_device.status,
            "model": model_name
        }
    
    # Coordinator returns full cluster status
    cluster_info = {
        "coordinator": local_device.device_id,
        "devices": {}
    }
    
    for device in DEVICES:
        if device.device_id in device_registry:
            info = device_registry[device.device_id]
            cluster_info["devices"][device.device_id] = {
                "hostname": device.hostname,
                "port": device.port,
                "role": device.role,
                "layers": device.layers,
                "status": device.status,
                "last_seen": info["last_seen"]
            }
        else:
            cluster_info["devices"][device.device_id] = {
                "hostname": device.hostname,
                "port": device.port,
                "role": device.role,
                "layers": device.layers,
                "status": "offline"
            }
    
    return cluster_info

@app.get("/v1/models")
async def list_models():
    """List available models"""
    if model_name:
        return {
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-distributed"
            }]
        }
    return {"data": []}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Handle chat completions using distributed inference"""
    global distributed_engine
    
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
        
        # Use distributed inference if available and we're the coordinator
        if distributed_engine and local_device.role == "coordinator":
            logger.info("🚀 Using distributed inference")
            
            # Create inference request
            inference_request = InferenceRequest(
                request_id=f"req_{int(time.time())}_{id(request)}",
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                model_name=request.model
            )
            
            # Process with distributed engine
            result = await distributed_engine.process_inference_request(inference_request)
            
            if result.success:
                logger.info(f"✅ Distributed inference completed: {result.total_tokens} tokens in {result.processing_time:.2f}s")
                
                return ChatResponse(
                    id=f"chatcmpl-{int(time.time())}",
                    created=int(time.time()),
                    model=model_name or request.model,
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.generated_text
                        },
                        "finish_reason": "stop"
                    }]
                )
            else:
                logger.error(f"Distributed inference failed: {result.error_message}")
                # Fall back to local generation
        
        # Fallback to local generation
        logger.info("📍 Using local generation fallback")
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded and distributed inference unavailable")
        
        if request.stream:
            return StreamingResponse(
                stream_response(prompt, request),
                media_type="text/event-stream"
            )
        else:
            # Create sampler with temperature and other parameters
            sampler = make_sampler(temp=request.temperature)
            
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=request.max_tokens,
                sampler=sampler
            )
            
            return ChatResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=model_name or request.model,
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
        
        # Create sampler with temperature
        sampler = make_sampler(temp=request.temperature)
        
        # Generate tokens using stream_generate
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler
        ):
            # Extract the token from the response object
            token = response.text if hasattr(response, 'text') else str(response)
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
        "device": local_device.device_id,
        "model": model_name,
        "timestamp": time.time()
    }

@app.get("/device-info")
async def device_info():
    """Get device information"""
    info = {
        "device_id": local_device.device_id,
        "hostname": local_device.hostname,
        "port": local_device.port,
        "role": local_device.role,
        "layers": local_device.layers,
        "mlx_device": str(mx.default_device()),
        "metal_available": mx.metal.is_available() if hasattr(mx, 'metal') else False,
        "model_loaded": model is not None,
        "model_name": model_name,
        "distributed_engine_ready": distributed_engine is not None
    }
    
    # Add distributed inference stats if available
    if distributed_engine and local_device.role == "coordinator":
        info["distributed_stats"] = distributed_engine.get_inference_stats()
    
    return info

@app.get("/distributed-stats")
async def get_distributed_stats():
    """Get distributed inference statistics (coordinator only)"""
    if not distributed_engine:
        raise HTTPException(status_code=503, detail="Distributed engine not available")
    
    if local_device.role != "coordinator":
        raise HTTPException(status_code=403, detail="Only coordinator can provide distributed stats")
    
    return distributed_engine.get_inference_stats()

if __name__ == "__main__":
    import sys
    
    # Determine port based on device
    local_device = get_local_device()
    port = local_device.port
    
    logger.info(f"Starting Distributed MLX Server")
    logger.info(f"Device: {local_device.device_id}")
    logger.info(f"Port: {port}")
    logger.info(f"MLX Device: {mx.default_device()}")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)