#!/usr/bin/env python3
"""
Enhanced Distributed MLX Inference Server
Integrates model splitting and improved tensor communication
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

# Import existing distributed components
from distributed_config import DistributedConfig, DeviceRole
from distributed_comm import DistributedCommunicator, CommunicationType
from distributed_mlx_inference import DistributedMLXInference

# Import new components
from model_splitter import ModelLayerSplitter, LAYER_ASSIGNMENTS
from grpc_tensor_service import start_grpc_server, GRPCTensorClient
from distributed_inference_new import DistributedInferenceEngine, InferenceRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "mlx-community/Qwen3-1.7B-8bit"
    messages: List[ChatMessage] = [{"role": "user", "content": "Hi!"}]
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
app = FastAPI(title="Enhanced Distributed MLX Server")

# Global state
config = None
distributed_engine = None
local_device_id = None
grpc_server_task = None

def get_local_device_id() -> str:
    """Determine local device ID based on hostname"""
    hostname = socket.gethostname().lower()
    
    if "mini1" in hostname:
        return "mini1"
    elif "mini2" in hostname:
        return "mini2"
    elif "master" in hostname:
        return "master"
    else:
        logger.warning(f"Unknown hostname {hostname}, defaulting to mini1")
        return "mini1"

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global config, distributed_engine, local_device_id, grpc_server_task
    
    try:
        # Load configuration
        config_path = os.getenv("DISTRIBUTED_CONFIG", "distributed_config.json")
        with open(config_path) as f:
            config_data = json.load(f)
        
        # Create configuration object
        config = DistributedConfig.from_dict(config_data)
        
        # Determine local device
        local_device_id = get_local_device_id()
        
        # Find device config by ID
        device_config = None
        for device in config.device_list:
            if device.device_id == local_device_id:
                device_config = device
                break
        
        if not device_config:
            raise ValueError(f"Device {local_device_id} not found in configuration")
        
        logger.info(f"🚀 Starting Enhanced Distributed MLX Server")
        logger.info(f"   Device: {local_device_id}")
        logger.info(f"   Role: {device_config.role.value}")
        logger.info(f"   MLX Device: {mx.default_device()}")
        
        # Determine layer assignment
        layer_assignment = next(
            (la for la in LAYER_ASSIGNMENTS if la.device_id == local_device_id),
            None
        )
        
        if layer_assignment:
            logger.info(f"   Layers: {layer_assignment.layers}")
        
        # Initialize distributed inference engine
        is_coordinator = (device_config.role == DeviceRole.MASTER)
        distributed_engine = DistributedInferenceEngine(local_device_id, is_coordinator)
        
        # Initialize in background
        asyncio.create_task(initialize_distributed_engine())
        
        # Start enhanced gRPC server for tensor communication
        if layer_assignment:
            grpc_port = 9100 + config.device_list.index(device_config)
            grpc_server_task = asyncio.create_task(
                start_grpc_server(
                    local_device_id,
                    device_config.hostname,
                    grpc_port,
                    layer_assignment.layers
                )
            )
            logger.info(f"   gRPC Server: Port {grpc_port}")
        
        logger.info("✅ Enhanced server initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

async def initialize_distributed_engine():
    """Initialize distributed inference engine"""
    global distributed_engine
    
    try:
        if distributed_engine:
            logger.info("Initializing distributed inference engine...")
            
            # Use the configured model
            model_name = f"{config.model.provider}/{config.model.name}"
            success = await distributed_engine.initialize(model_name)
            
            if success:
                logger.info("✅ Distributed inference engine ready")
            else:
                logger.error("❌ Failed to initialize distributed inference engine")
                
    except Exception as e:
        logger.error(f"Error initializing distributed engine: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Distributed MLX Server",
        "device": local_device_id,
        "model": f"{config.model.provider}/{config.model.name}" if config else None,
        "status": "ready" if distributed_engine else "initializing"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": local_device_id,
        "timestamp": time.time(),
        "model_loaded": distributed_engine is not None,
        "mlx_device": str(mx.default_device())
    }

@app.get("/device-info")
async def device_info():
    """Get detailed device information"""
    info = {
        "device_id": local_device_id,
        "hostname": socket.gethostname(),
        "mlx_device": str(mx.default_device()),
        "metal_available": mx.metal.is_available() if hasattr(mx, 'metal') else False,
        "distributed_engine_ready": distributed_engine is not None
    }
    
    # Add device capabilities if available
    if config:
        device_config = config.get_device(local_device_id)
        if device_config and hasattr(device_config, 'capabilities'):
            info["capabilities"] = device_config.capabilities
    
    # Add distributed stats if coordinator
    if distributed_engine and hasattr(distributed_engine, 'get_inference_stats'):
        info["distributed_stats"] = distributed_engine.get_inference_stats()
    
    return info

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Handle chat completions with distributed inference"""
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
        
        logger.info(f"Processing chat completion: {len(request.messages)} messages")
        
        # Use distributed inference if available and we're the coordinator
        if distributed_engine and config and config.get_device(local_device_id).role == "coordinator":
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
                    model=request.model,
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
                raise HTTPException(status_code=500, detail=result.error_message)
        else:
            # Not coordinator or distributed engine not ready
            if config and config.get_device(local_device_id).role != "coordinator":
                raise HTTPException(
                    status_code=403, 
                    detail=f"This device ({local_device_id}) is not the coordinator. Send requests to the coordinator."
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Distributed inference engine not ready"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/cluster-status")
async def cluster_status():
    """Get cluster status"""
    if not config:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    
    status = {
        "coordinator": config.get_master_device().device_id if config.get_master_device() else "unknown",
        "devices": []
    }
    
    for device in config.device_list:
        device_info = {
            "device_id": device.device_id,
            "hostname": device.hostname,
            "port": device.port,
            "role": device.role.value,
            "status": "unknown"
        }
        
        # Check if this is the local device
        if device.device_id == local_device_id:
            device_info["status"] = "online"
            device_info["is_local"] = True
        
        status["devices"].append(device_info)
    
    return status

@app.get("/gpu-stats")
async def gpu_stats():
    """Get GPU utilization statistics"""
    try:
        # Try to get GPU stats using subprocess
        import subprocess
        
        # For Apple Silicon, we can try to get some basic info
        gpu_info = {
            "device": str(mx.default_device()),
            "metal_available": mx.metal.is_available() if hasattr(mx, 'metal') else False,
            "timestamp": time.time()
        }
        
        # Try to get more detailed stats if available
        try:
            # This is a placeholder - actual GPU monitoring would need platform-specific tools
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                import json
                display_data = json.loads(result.stdout)
                gpu_info["system_info"] = display_data
        except:
            pass
        
        return gpu_info
        
    except Exception as e:
        logger.error(f"Failed to get GPU stats: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import sys
    
    # Get port from environment or use default
    port = int(os.getenv("API_PORT", 8100))
    
    logger.info(f"Starting Enhanced Distributed MLX Server on port {port}")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)