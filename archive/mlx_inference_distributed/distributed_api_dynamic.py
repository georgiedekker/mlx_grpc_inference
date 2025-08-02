#!/usr/bin/env python3
"""
Dynamic distributed API server with automatic device discovery.
Drop-in replacement for the static distributed_api.py.
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dynamic_cluster_manager import DynamicClusterManager
from distributed_mlx_inference import DistributedMLXInference
from distributed_comm import create_communicator, CommunicationBackend
from mlx_discovery import MLXServiceDiscovery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MLX Distributed Inference API (Dynamic)",
    description="OpenAI-compatible API with automatic device discovery",
    version="2.0.0"
)

# Global state
cluster_manager: Optional[DynamicClusterManager] = None
communicator = None
distributed_inference = None

# API Models (same as original)
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@app.on_event("startup")
async def startup_event():
    """Initialize the dynamic distributed system on startup."""
    global cluster_manager, communicator, distributed_inference
    
    logger.info("🚀 Starting MLX Distributed API with dynamic discovery...")
    
    # Get model from environment or use default
    model_id = os.environ.get("MLX_MODEL_ID", "mlx-community/Qwen3-1.7B-8bit")
    master_port = int(os.environ.get("MLX_MASTER_PORT", "50100"))
    
    # Create dynamic cluster manager
    cluster_manager = DynamicClusterManager(
        model_id=model_id,
        master_port=master_port
    )
    
    # Start discovery and wait for initial devices
    cluster_manager.start()
    
    # Wait a bit for devices to be discovered
    await asyncio.sleep(3)
    
    # Get initial cluster configuration
    config = cluster_manager.get_cluster_config()
    logger.info(f"📊 Initial cluster: {config.world_size} devices discovered")
    
    # Initialize communicator
    communicator = create_communicator(CommunicationBackend.GRPC)
    device_hostnames = [d.hostname for d in config.devices]
    communicator.init(
        rank=0,  # API server is always rank 0
        world_size=config.world_size,
        device_hostnames=device_hostnames
    )
    
    # Initialize distributed inference
    distributed_inference = DistributedMLXInference(config, communicator)
    
    logger.info(f"✅ API server ready with {config.world_size} devices")
    logger.info(f"📡 Devices will be automatically discovered and added to the cluster")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if cluster_manager:
        cluster_manager.stop()
    if communicator:
        communicator.finalize()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "MLX Distributed Inference API (Dynamic)",
        "version": "2.0.0",
        "features": [
            "Automatic device discovery",
            "Dynamic shard rebalancing",
            "Thunderbolt support",
            "Maximum RAM utilization"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": distributed_inference is not None,
        "model_name": cluster_manager.model_id if cluster_manager else None,
        "cluster_size": len(cluster_manager.cluster_state.active_workers) if cluster_manager else 0,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/distributed/cluster-info")
async def cluster_info():
    """Get current cluster information with dynamic devices."""
    if not cluster_manager:
        raise HTTPException(status_code=503, detail="Cluster manager not initialized")
        
    config = cluster_manager.get_cluster_config()
    
    # Get Thunderbolt connectivity
    tb_connectivity = cluster_manager.check_thunderbolt_connectivity()
    
    return {
        "cluster_info": {
            "total_devices": config.world_size,
            "total_memory_gb": cluster_manager.cluster_state.total_memory_gb,
            "total_gpu_cores": cluster_manager.cluster_state.total_gpu_cores,
            "discovery_enabled": True,
            "auto_rebalancing": True,
            "thunderbolt_devices": list(tb_connectivity.keys())
        },
        "devices": [
            {
                "device_id": d.device_id,
                "hostname": d.hostname,
                "port": d.port,
                "role": d.role.value,
                "memory_gb": d.capabilities.get('memory_gb', 0),
                "gpu_cores": d.capabilities.get('gpu_cores', 0),
                "status": "online"
            }
            for d in config.devices
        ],
        "sharding_plan": cluster_manager.cluster_state.current_sharding.dict() if cluster_manager.cluster_state.current_sharding else None
    }

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": cluster_manager.model_id if cluster_manager else "unknown",
                "object": "model",
                "created": 1677610602,
                "owned_by": "mlx-distributed",
                "permission": [],
                "root": cluster_manager.model_id if cluster_manager else "unknown",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using distributed inference."""
    if not distributed_inference:
        raise HTTPException(status_code=503, detail="Inference engine not ready")
        
    # Check if we have any workers
    if cluster_manager.cluster_state.total_gpu_cores == 0:
        raise HTTPException(
            status_code=503, 
            detail="No worker devices available. Please start worker_simple.py on other devices."
        )
        
    # Convert messages to format expected by inference engine
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    try:
        # Update inference engine with latest cluster config if needed
        current_config = cluster_manager.get_cluster_config()
        if current_config.world_size != distributed_inference.config.world_size:
            logger.info(f"🔄 Updating inference engine for {current_config.world_size} devices")
            # In a full implementation, we'd reinitialize here
            
        # Perform distributed inference
        response, token_count = distributed_inference.chat(
            messages=messages,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=1.1,
            return_token_count=True
        )
        
        # Format response
        import time
        completion_id = f"chatcmpl-{int(time.time())}"
        
        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": len(messages[0]["content"].split()),  # Rough estimate
                "completion_tokens": token_count,
                "total_tokens": len(messages[0]["content"].split()) + token_count
            }
        )
        
    except Exception as e:
        logger.error(f"Distributed inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/distributed/add-device")
async def add_device_hint(hostname: str):
    """Hint to check for a specific device (useful for Thunderbolt direct connections)."""
    # In a full implementation, this would actively probe the device
    return {"message": f"Will check for device at {hostname}"}

def main():
    """Main entry point."""
    port = int(os.environ.get("API_PORT", "8100"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()