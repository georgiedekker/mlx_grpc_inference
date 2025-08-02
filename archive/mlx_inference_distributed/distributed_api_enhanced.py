#!/usr/bin/env python3
"""
Enhanced distributed API server with dynamic discovery and improved inference.
Production-ready implementation with all requested features.
"""

import os
import sys
import json
import logging
import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dynamic_cluster_manager import DynamicClusterManager
from distributed_mlx_inference_dynamic import DistributedMLXInferenceDynamic
from distributed_comm import create_communicator, CommunicationBackend
from distributed_config import DistributedConfig, DeviceConfig, DeviceRole
from sharding_strategy import ShardInfo
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MLX Distributed Inference API (Enhanced)",
    description="Production-ready API with dynamic discovery and optimized inference",
    version="3.0.0"
)

# Global state
cluster_manager: Optional[DynamicClusterManager] = None
communicator = None
distributed_inference: Optional[DistributedMLXInferenceDynamic] = None
initialization_lock = threading.Lock()
last_cluster_size = 0

# API Models
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
    performance: Optional[Dict[str, Any]] = None  # Added performance metrics

async def reinitialize_inference_engine():
    """Reinitialize inference engine when cluster changes."""
    global distributed_inference, communicator, last_cluster_size
    
    with initialization_lock:
        config = cluster_manager.get_cluster_config()
        
        if config.world_size == 0:
            logger.warning("No devices available for inference")
            # For testing, allow single device operation
            config = DistributedConfig(
                model_name=cluster_manager.model_id,
                device_list=[DeviceConfig(
                    device_id="mini1",
                    hostname="mini1.local", 
                    port=50100,
                    role=DeviceRole.MASTER,
                    device_index=0,
                    capabilities={'memory_gb': 16, 'gpu_cores': 10}
                )],
                model_parallel_size=1
            )
            config.world_size = 1
            config.devices = config.device_list
            
        if config.world_size == last_cluster_size:
            # Just update shards if cluster size hasn't changed
            if hasattr(cluster_manager.cluster_state, 'shard_updates'):
                updates = cluster_manager.cluster_state.shard_updates
                if distributed_inference and 'mini1' in updates:
                    distributed_inference.update_shard_assignment(updates['mini1'])
            return
            
        logger.info(f"🔄 Reinitializing inference engine for {config.world_size} devices")
        
        # Shutdown old communicator
        if communicator:
            try:
                communicator.finalize()
            except:
                pass
                
        # Create new communicator
        communicator = create_communicator(CommunicationBackend.GRPC)
        device_hostnames = [d.hostname for d in config.devices]
        communicator.init(
            rank=0,
            world_size=config.world_size,
            device_hostnames=device_hostnames
        )
        
        # Create new inference engine
        distributed_inference = DistributedMLXInferenceDynamic(config, communicator)
        
        # Load model shard for master
        if hasattr(cluster_manager.cluster_state, 'current_sharding'):
            plan = cluster_manager.cluster_state.current_sharding
            if plan and 'mini1' in plan.device_assignments:
                assignment = plan.device_assignments['mini1']
                shard_info = ShardInfo(
                    start_layer=assignment['start_layer'],
                    end_layer=assignment['end_layer'],
                    total_layers=assignment['total_layers'],
                    has_embeddings=True,  # Master always has embeddings
                    has_head=(assignment['end_layer'] == assignment['total_layers'] - 1)
                )
                distributed_inference.load_model_shard(shard_info)
                
        last_cluster_size = config.world_size
        logger.info(f"✅ Inference engine ready with {config.world_size} devices")

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced distributed system on startup."""
    global cluster_manager
    
    logger.info("🚀 Starting MLX Enhanced Distributed API...")
    
    # Get configuration
    model_id = os.environ.get("MLX_MODEL_ID", "mlx-community/Qwen3-1.7B-8bit")
    master_port = int(os.environ.get("MLX_MASTER_PORT", "50100"))
    
    # Create dynamic cluster manager
    cluster_manager = DynamicClusterManager(
        model_id=model_id,
        master_port=master_port
    )
    
    # Start discovery in a thread to avoid blocking
    def start_cluster_manager():
        cluster_manager.start()
    
    threading.Thread(target=start_cluster_manager, daemon=True).start()
    await asyncio.sleep(1)  # Give it time to start
    
    # Set up callbacks for cluster changes after starting
    if cluster_manager.discovery:
        original_add = cluster_manager.discovery.on_device_added
        original_remove = cluster_manager.discovery.on_device_removed
        
        async def on_device_change(*args):
            """Handle device changes by reinitializing inference engine."""
            await asyncio.sleep(2)  # Wait for cluster to stabilize
            await reinitialize_inference_engine()
            
        # Wrap callbacks
        if original_add:
            def new_add(worker):
                original_add(worker)
                asyncio.create_task(on_device_change())
            cluster_manager.discovery.on_device_added = new_add
            
        if original_remove:
            def new_remove(hostname):
                original_remove(hostname)
                asyncio.create_task(on_device_change())
            cluster_manager.discovery.on_device_removed = new_remove
    
    # Wait for initial devices
    await asyncio.sleep(3)
    
    # Initialize inference engine
    await reinitialize_inference_engine()
    
    logger.info("✅ Enhanced API server ready")
    logger.info("📡 Devices will be automatically discovered and added")
    logger.info("🔄 Model shards will be dynamically rebalanced")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if distributed_inference:
        distributed_inference.shutdown()
    if cluster_manager:
        cluster_manager.stop()
    if communicator:
        communicator.finalize()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "MLX Distributed Inference API (Enhanced)",
        "version": "3.0.0",
        "features": [
            "Automatic device discovery via mDNS",
            "Dynamic shard rebalancing",
            "Thunderbolt network detection",
            "Maximum RAM utilization",
            "Asynchronous tensor passing",
            "No barriers or synchronization",
            "Partial model loading",
            "Real-time performance metrics"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status."""
    memory_stats = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "model_loaded": distributed_inference is not None and distributed_inference._model_loaded,
        "model_name": cluster_manager.model_id if cluster_manager else None,
        "cluster_size": len(cluster_manager.cluster_state.active_workers) if cluster_manager else 0,
        "total_memory_gb": cluster_manager.cluster_state.total_memory_gb if cluster_manager else 0,
        "total_gpu_cores": cluster_manager.cluster_state.total_gpu_cores if cluster_manager else 0,
        "host_memory": {
            "total_gb": memory_stats.total / (1024**3),
            "available_gb": memory_stats.available / (1024**3),
            "percent_used": memory_stats.percent
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/distributed/cluster-info")
async def cluster_info():
    """Get detailed cluster information."""
    if not cluster_manager:
        raise HTTPException(status_code=503, detail="Cluster manager not initialized")
        
    config = cluster_manager.get_cluster_config()
    tb_connectivity = cluster_manager.check_thunderbolt_connectivity()
    
    # Get memory usage from inference engine
    memory_usage = {}
    if distributed_inference:
        memory_usage = distributed_inference.get_memory_usage()
    
    # Calculate theoretical tokens/sec based on hardware
    total_gpu_cores = cluster_manager.cluster_state.total_gpu_cores
    estimated_tps = total_gpu_cores * 1.5  # Rough estimate: 1.5 tokens/sec per GPU core
    
    return {
        "cluster_info": {
            "total_devices": config.world_size,
            "total_memory_gb": cluster_manager.cluster_state.total_memory_gb,
            "total_gpu_cores": total_gpu_cores,
            "estimated_tokens_per_sec": estimated_tps,
            "discovery_enabled": True,
            "auto_rebalancing": True,
            "thunderbolt_devices": list(tb_connectivity.keys()),
            "inference_engine": "dynamic_async",
            "optimizations": [
                "Partial model loading",
                "Asynchronous tensor passing",
                "Memory-proportional sharding",
                "No synchronization barriers"
            ]
        },
        "devices": [
            {
                "device_id": d.device_id,
                "hostname": d.hostname,
                "port": d.port,
                "role": d.role.value,
                "memory_gb": d.capabilities.get('memory_gb', 0),
                "available_memory_gb": d.capabilities.get('available_memory_gb', 0),
                "gpu_cores": d.capabilities.get('gpu_cores', 0),
                "model": d.capabilities.get('model', 'Unknown'),
                "status": "online"
            }
            for d in config.devices
        ],
        "sharding_plan": cluster_manager.cluster_state.current_sharding.dict() if cluster_manager.cluster_state.current_sharding else None,
        "memory_usage": memory_usage
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
                "owned_by": "mlx-distributed-enhanced",
                "permission": [],
                "root": cluster_manager.model_id if cluster_manager else "unknown",
                "parent": None,
                "capabilities": {
                    "max_context_length": 8192,
                    "distributed": True,
                    "dynamic_sharding": True
                }
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using enhanced distributed inference."""
    if not distributed_inference or not distributed_inference._model_loaded:
        # Try to reinitialize
        await reinitialize_inference_engine()
        
        if not distributed_inference or not distributed_inference._model_loaded:
            raise HTTPException(
                status_code=503, 
                detail="Inference engine not ready. Waiting for devices to join cluster."
            )
        
    # Check cluster health
    if cluster_manager.cluster_state.total_gpu_cores == 0:
        raise HTTPException(
            status_code=503, 
            detail="No worker devices available. Start worker_simple.py on other devices."
        )
        
    # Convert messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Track performance
    start_time = time.time()
    inference_start = None
    
    try:
        # Log request
        logger.info(f"Processing chat completion: {len(messages)} messages, max_tokens={request.max_tokens}")
        
        # Perform distributed inference
        inference_start = time.time()
        response, token_count = distributed_inference.chat(
            messages=messages,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=1.1,
            return_token_count=True
        )
        inference_time = time.time() - inference_start
        
        # Calculate metrics
        total_time = time.time() - start_time
        tokens_per_second = token_count / inference_time if inference_time > 0 else 0
        
        # Get prompt token estimate
        prompt_text = " ".join(msg["content"] for msg in messages)
        prompt_tokens = len(prompt_text.split()) * 1.5  # Rough estimate
        
        # Format response
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        response_data = ChatCompletionResponse(
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
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": token_count,
                "total_tokens": int(prompt_tokens) + token_count
            },
            performance={
                "inference_time_sec": round(inference_time, 3),
                "total_time_sec": round(total_time, 3),
                "tokens_per_second": round(tokens_per_second, 2),
                "devices_used": cluster_manager.cluster_state.total_gpu_cores // 10,  # Rough device count
                "total_gpu_cores": cluster_manager.cluster_state.total_gpu_cores
            }
        )
        
        logger.info(f"Completed: {token_count} tokens in {inference_time:.2f}s ({tokens_per_second:.1f} TPS)")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Distributed inference failed: {e}", exc_info=True)
        
        # Provide helpful error message
        if "Timeout" in str(e):
            detail = "Inference timeout - workers may be offline or overloaded"
        elif "None logits" in str(e):
            detail = "Tensor passing failed - check worker connectivity"
        else:
            detail = f"Inference error: {str(e)}"
            
        raise HTTPException(status_code=500, detail=detail)

@app.get("/distributed/performance")
async def get_performance_metrics():
    """Get detailed performance metrics."""
    if not distributed_inference:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
        
    # Get inference times
    recent_times = distributed_inference.inference_times[-10:] if distributed_inference.inference_times else []
    comm_times = distributed_inference.communication_times[-10:] if distributed_inference.communication_times else []
    
    # Calculate averages
    avg_inference = sum(recent_times) / len(recent_times) if recent_times else 0
    avg_comm = sum(comm_times) / len(comm_times) if comm_times else 0
    
    return {
        "inference_metrics": {
            "recent_inference_times": recent_times,
            "average_inference_time": round(avg_inference, 3),
            "recent_communication_times": comm_times,
            "average_communication_time": round(avg_comm, 3),
            "communication_overhead_percent": round((avg_comm / avg_inference * 100) if avg_inference > 0 else 0, 1)
        },
        "cluster_metrics": {
            "active_devices": cluster_manager.cluster_state.total_gpu_cores // 10,
            "total_memory_gb": cluster_manager.cluster_state.total_memory_gb,
            "memory_utilization_percent": 0  # TODO: Calculate actual usage
        }
    }

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