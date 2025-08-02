"""
Distributed FastAPI server for MLX inference.

This module provides the API server that coordinates distributed inference
across multiple devices. It handles request routing, load balancing, and
aggregation of results from worker nodes.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal, AsyncGenerator
import asyncio
import aiohttp
import json
import time
import uuid
import logging
from datetime import datetime
import os

from distributed_config import DistributedConfig, DeviceRole, DeviceConfig
from distributed_mlx_inference import DistributedMLXInference
from distributed_comm import create_communicator, CommunicationBackend
from openai_api import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatMessage, CompletionRequest, CompletionResponse, CompletionChoice,
    Model, ModelList
)

logger = logging.getLogger(__name__)

# App will be created after lifespan is defined

# Global state
distributed_inference: Optional[DistributedMLXInference] = None
config: Optional[DistributedConfig] = None
communicator = None
device_health: Dict[str, Dict[str, Any]] = {}


class DistributedStatusResponse(BaseModel):
    """Response for distributed system status."""
    status: str
    role: str
    device_id: str
    rank: int
    world_size: int
    model_loaded: bool
    model_name: str
    devices: List[Dict[str, Any]]
    performance_stats: Optional[Dict[str, Any]] = None


class DeviceHealthCheck(BaseModel):
    """Health check information for a device."""
    device_id: str
    hostname: str
    status: str
    last_heartbeat: float
    response_time: Optional[float] = None
    error: Optional[str] = None


async def check_device_health(device: DeviceConfig) -> DeviceHealthCheck:
    """Check health of a remote device."""
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://{device.hostname}:{device.port}/health"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    response_time = time.time() - start_time
                    return DeviceHealthCheck(
                        device_id=device.device_id,
                        hostname=device.hostname,
                        status="healthy",
                        last_heartbeat=time.time(),
                        response_time=response_time
                    )
                else:
                    return DeviceHealthCheck(
                        device_id=device.device_id,
                        hostname=device.hostname,
                        status="unhealthy",
                        last_heartbeat=time.time(),
                        error=f"HTTP {response.status}"
                    )
    except asyncio.TimeoutError:
        return DeviceHealthCheck(
            device_id=device.device_id,
            hostname=device.hostname,
            status="timeout",
            last_heartbeat=time.time(),
            error="Connection timeout"
        )
    except Exception as e:
        return DeviceHealthCheck(
            device_id=device.device_id,
            hostname=device.hostname,
            status="error",
            last_heartbeat=time.time(),
            error=str(e)
        )


async def monitor_device_health():
    """Background task to monitor device health."""
    global device_health
    
    while True:
        if config:
            tasks = []
            for device in config.device_list:
                if device.device_id != config.get_device_by_index(get_local_rank()).device_id:
                    tasks.append(check_device_health(device))
            
            if tasks:
                results = await asyncio.gather(*tasks)
                for result in results:
                    device_health[result.device_id] = result.model_dump()
        
        await asyncio.sleep(config.heartbeat_interval if config else 5.0)


def get_local_rank() -> int:
    """Get local device rank from environment or config."""
    # Get rank from environment variable
    # This can be set by the launch script or container orchestrator
    return int(os.environ.get("LOCAL_RANK", "0"))


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

app = FastAPI(title="Distributed MLX OpenAI-Compatible API", version="2.0.0", lifespan=lifespan)

async def startup_event():
    """Initialize distributed system on startup."""
    global distributed_inference, config, communicator
    
    try:
        # Load configuration
        config_path = os.environ.get("DISTRIBUTED_CONFIG", "distributed_config.json")
        if os.path.exists(config_path):
            config = DistributedConfig.load(config_path)
        else:
            config = DistributedConfig()
            config.save(config_path)
        
        # Get local rank
        local_rank = get_local_rank()
        
        # Create communicator
        backend = CommunicationBackend(config.communication_backend)
        communicator = create_communicator(backend)
        
        # Get world size from config
        world_size = config.model_parallel_size
        
        # Collect device hostnames for gRPC communication
        device_hostnames = []
        for i in range(world_size):
            device = config.get_device_by_index(i)
            if device:
                # Use hostname without the API port (e.g., "mini2.local" instead of "mini2.local:8001")
                hostname = device.hostname.split(':')[0] if ':' in device.hostname else device.hostname
                device_hostnames.append(hostname)
            else:
                device_hostnames.append("localhost")
        
        # Initialize gRPC communicator (use separate port range for communication)
        communicator.init(rank=local_rank, world_size=world_size, device_hostnames=device_hostnames)
        
        # Initialize distributed inference
        distributed_inference = DistributedMLXInference(
            config=config,
            communicator=communicator,
            local_rank=local_rank
        )
        
        # Start health monitoring for master node
        if config.get_device_by_index(local_rank).role == DeviceRole.MASTER:
            asyncio.create_task(monitor_device_health())
        
        device_info = config.get_device_by_index(local_rank)
        logger.info(f"Distributed inference initialized:")
        logger.info(f"  - Rank: {local_rank}")
        logger.info(f"  - World size: {world_size}")
        logger.info(f"  - Role: {device_info.role if device_info else 'unknown'}")
        logger.info(f"  - Device ID: {device_info.device_id if device_info else 'unknown'}")
        logger.info(f"  - gRPC initialized: {communicator._initialized if hasattr(communicator, '_initialized') else 'unknown'}")
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed system: {str(e)}")
        raise


async def shutdown_event():
    """Cleanup on shutdown."""
    global communicator
    
    if communicator:
        communicator.finalize()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    local_rank = get_local_rank()
    device = config.get_device_by_index(local_rank) if config else None
    
    return {
        "message": "Distributed MLX Inference API",
        "status": "running",
        "device_id": device.device_id if device else "unknown",
        "role": device.role.value if device else "unknown"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    local_rank = get_local_rank()
    device = config.get_device_by_index(local_rank) if config else None
    
    return {
        "status": "healthy" if distributed_inference else "unhealthy",
        "model_loaded": distributed_inference is not None,
        "model_name": config.model_name if config else "unknown",
        "device_id": device.device_id if device else "unknown",
        "role": device.role.value if device else "unknown",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/test/simple")
async def test_simple():
    """Simple test endpoint that doesn't use distributed inference."""
    return {
        "status": "ok",
        "rank": get_local_rank(),
        "device_id": config.get_device_by_index(get_local_rank()).device_id if config else "unknown",
        "grpc_initialized": communicator._initialized if communicator and hasattr(communicator, '_initialized') else False,
        "inference_loaded": distributed_inference is not None
    }


@app.get("/distributed/gpu-info")
async def gpu_info():
    """Get GPU information and status from all devices in the cluster."""
    import mlx.core as mx
    
    local_rank = get_local_rank()
    device = config.get_device_by_index(local_rank) if config else None
    
    # Get local GPU info with hardware capabilities
    def get_local_gpu_info(device_id: str, rank: int):
        # Test GPU performance
        test_size = 1000
        a = mx.random.normal([test_size, test_size])
        b = mx.random.normal([test_size, test_size])
        
        import time
        start = time.time()
        c = a @ b
        mx.eval(c)
        compute_time = time.time() - start
        
        # Get hardware capabilities from device config or JSON file
        capabilities = {}
        if device and hasattr(device, 'capabilities') and device.capabilities:
            capabilities = device.capabilities.copy()
        else:
            # Try to load capabilities from JSON config file
            try:
                import json
                config_path = os.environ.get("DISTRIBUTED_CONFIG", "distributed_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        json_config = json.load(f)
                    
                    # Find this device in the JSON config
                    for json_device in json_config.get("devices", []):
                        if json_device.get("device_id") == device_id:
                            capabilities = json_device.get("capabilities", {})
                            break
            except Exception as e:
                logger.debug(f"Could not load capabilities from JSON: {e}")
        
        # Get system memory info using psutil
        import psutil
        memory = psutil.virtual_memory()
        
        base_info = {
            "device_id": device_id,
            "rank": rank,
            "role": device.role.value if device else "unknown",
            "hostname": device.hostname if device else "unknown",
            "mlx_default_device": str(mx.default_device()),
            "performance_test": {
                "computation_time": compute_time,
                "test_size": f"{test_size}x{test_size} matrix multiplication",
                "status": "GPU active" if compute_time < 0.1 else "Possibly using CPU",
                "throughput_gflops": round((2 * test_size**3) / (compute_time * 1e9), 2) if compute_time > 0 else None
            },
            "system_memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent
            }
        }
        
        # Add hardware capabilities if available
        if capabilities:
            base_info["hardware"] = {
                "model": capabilities.get("model", "Unknown"),
                "memory_gb": capabilities.get("memory_gb", None),
                "gpu_cores": capabilities.get("gpu_cores", None),
                "cpu_cores": capabilities.get("cpu_cores", None),
                "cpu_performance_cores": capabilities.get("cpu_performance_cores", None),
                "cpu_efficiency_cores": capabilities.get("cpu_efficiency_cores", None),
                "neural_engine_cores": capabilities.get("neural_engine_cores", None),
                "bandwidth_gbps": capabilities.get("bandwidth_gbps", None),
                "mlx_metal_available": capabilities.get("mlx_metal_available", True),
                "max_recommended_model_size_gb": capabilities.get("max_recommended_model_size_gb", None)
            }
        
        return base_info
    
    # If we're the master, collect info from all devices
    if device and device.role == DeviceRole.MASTER and config:
        devices_info = []
        
        # Add local device info
        local_info = get_local_gpu_info(device.device_id, local_rank)
        devices_info.append(local_info)
        
        # Add info for other devices (based on configuration)
        for i in range(config.model_parallel_size):
            if i != local_rank:
                other_device = config.get_device_by_index(i)
                if other_device:
                    # Get capabilities from config for worker devices
                    capabilities = {}
                    if hasattr(other_device, 'capabilities') and other_device.capabilities:
                        capabilities = other_device.capabilities.copy()
                    else:
                        # Try to load capabilities from JSON config file
                        try:
                            import json
                            config_path = os.environ.get("DISTRIBUTED_CONFIG", "distributed_config.json")
                            if os.path.exists(config_path):
                                with open(config_path, 'r') as f:
                                    json_config = json.load(f)
                                
                                # Find this device in the JSON config
                                for json_device in json_config.get("devices", []):
                                    if json_device.get("device_id") == other_device.device_id:
                                        capabilities = json_device.get("capabilities", {})
                                        break
                        except Exception as e:
                            logger.debug(f"Could not load capabilities from JSON for {other_device.device_id}: {e}")
                    
                    device_info = {
                        "device_id": other_device.device_id,
                        "rank": i,
                        "role": other_device.role.value,
                        "hostname": other_device.hostname,
                        "mlx_default_device": "Device(gpu, 0)",  # Assume GPU for Apple Silicon
                        "performance_test": {
                            "computation_time": None,  # Can't test remotely without API
                            "test_size": "N/A (worker node)",
                            "status": "Worker - GPU assumed active",
                            "throughput_gflops": None
                        },
                        "system_memory": {
                            "total_gb": capabilities.get("memory_gb", None),
                            "available_gb": None,  # Can't get real-time info remotely
                            "used_percent": None
                        },
                        "note": "Worker node - live performance metrics not available via gRPC"
                    }
                    
                    # Add hardware capabilities if available
                    if capabilities:
                        device_info["hardware"] = {
                            "model": capabilities.get("model", "Unknown"),
                            "memory_gb": capabilities.get("memory_gb", None),
                            "gpu_cores": capabilities.get("gpu_cores", None),
                            "cpu_cores": capabilities.get("cpu_cores", None),
                            "cpu_performance_cores": capabilities.get("cpu_performance_cores", None),
                            "cpu_efficiency_cores": capabilities.get("cpu_efficiency_cores", None),
                            "neural_engine_cores": capabilities.get("neural_engine_cores", None),
                            "bandwidth_gbps": capabilities.get("bandwidth_gbps", None),
                            "mlx_metal_available": capabilities.get("mlx_metal_available", True),
                            "max_recommended_model_size_gb": capabilities.get("max_recommended_model_size_gb", None)
                        }
                    
                    devices_info.append(device_info)
        
        # Calculate cluster totals
        total_gpu_cores = sum(d.get("hardware", {}).get("gpu_cores", 0) for d in devices_info if d.get("hardware", {}).get("gpu_cores"))
        total_cpu_cores = sum(d.get("hardware", {}).get("cpu_cores", 0) for d in devices_info if d.get("hardware", {}).get("cpu_cores"))
        total_memory_gb = sum(d.get("hardware", {}).get("memory_gb", 0) for d in devices_info if d.get("hardware", {}).get("memory_gb"))
        total_neural_engine_cores = sum(d.get("hardware", {}).get("neural_engine_cores", 0) for d in devices_info if d.get("hardware", {}).get("neural_engine_cores"))
        
        # Get unique models in cluster
        models = list(set(d.get("hardware", {}).get("model", "Unknown") for d in devices_info if d.get("hardware", {}).get("model")))
        
        return {
            "cluster_info": {
                "total_devices": len(devices_info),
                "healthy_devices": len(devices_info),  # All devices assumed healthy if in config
                "world_size": config.model_parallel_size,
                "gRPC_communication": "Active" if communicator and communicator._initialized else "Inactive",
                "aggregate_hardware": {
                    "total_gpu_cores": total_gpu_cores if total_gpu_cores > 0 else None,
                    "total_cpu_cores": total_cpu_cores if total_cpu_cores > 0 else None,
                    "total_memory_gb": total_memory_gb if total_memory_gb > 0 else None,
                    "total_neural_engine_cores": total_neural_engine_cores if total_neural_engine_cores > 0 else None,
                    "device_models": models if models else ["Unknown"]
                }
            },
            "devices": devices_info
        }
    else:
        # Non-master devices return only local info
        return get_local_gpu_info(device.device_id if device else "unknown", local_rank)


@app.get("/distributed/status", response_model=DistributedStatusResponse)
async def distributed_status():
    """Get detailed distributed system status."""
    local_rank = get_local_rank()
    device = config.get_device_by_index(local_rank) if config else None
    
    # Collect device information
    devices_info = []
    for dev in config.device_list if config else []:
        dev_info = {
            "device_id": dev.device_id,
            "hostname": dev.hostname,
            "port": dev.port,
            "role": dev.role.value,
            "rank": dev.device_index
        }
        
        # Add health info if available
        if dev.device_id in device_health:
            dev_info.update(device_health[dev.device_id])
        elif dev.device_id == device.device_id:
            dev_info.update({
                "status": "healthy",
                "last_heartbeat": time.time()
            })
        
        devices_info.append(dev_info)
    
    # Get performance stats if available
    perf_stats = None
    if distributed_inference:
        perf_stats = distributed_inference.get_performance_stats()
    
    return DistributedStatusResponse(
        status="operational" if distributed_inference else "initializing",
        role=device.role.value if device else "unknown",
        device_id=device.device_id if device else "unknown",
        rank=local_rank,
        world_size=config.model_parallel_size if config else 0,
        model_loaded=distributed_inference is not None,
        model_name=config.model_name if config else "unknown",
        devices=devices_info,
        performance_stats=perf_stats
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion using distributed inference.
    
    Only the master node should receive external API requests.
    """
    if not distributed_inference:
        raise HTTPException(status_code=503, detail="Distributed inference not initialized")
    
    local_rank = get_local_rank()
    device = config.get_device_by_index(local_rank)
    
    # Only master should handle external requests
    if device.role != DeviceRole.MASTER:
        raise HTTPException(
            status_code=403, 
            detail="Only master node can handle chat completion requests"
        )
    
    try:
        # Convert messages to the format expected by our inference module
        messages = []
        for msg in request.messages:
            if msg.content:
                messages.append({"role": msg.role, "content": msg.content})
        
        if not messages:
            raise ValueError("No valid messages provided")
        
        # Handle streaming (not implemented in distributed version yet)
        if request.stream:
            raise HTTPException(
                status_code=501, 
                detail="Streaming not yet supported in distributed mode"
            )
        
        # Perform distributed inference
        logger.info(f"Starting distributed inference with {len(messages)} messages")
        
        # Note: The distributed inference coordination happens inside the 
        # distributed_inference.chat() method through gRPC calls
        
        try:
            response, token_count = distributed_inference.chat(
                messages=messages,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=1.1,  # Convert from frequency_penalty
                return_token_count=True
            )
            logger.info(f"Distributed inference completed: {token_count} tokens")
        except Exception as e:
            logger.error(f"Distributed inference failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Estimate prompt tokens
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        prompt_tokens = len(prompt_text.split()) * 2  # Rough estimate
        
        return ChatCompletionResponse(
            model=config.model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response),
                    finish_reason="stop" if token_count < (request.max_tokens or 512) else "length"
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Distributed inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models."""
    model_id = config.model_name if config else "unknown"
    
    return ModelList(
        data=[
            Model(
                id=model_id,
                created=int(time.time()),
                owned_by="mlx-community",
                root=model_id,
                permission=[]
            )
        ]
    )


# Worker-specific endpoints (for internal communication)
@app.post("/internal/forward")
async def internal_forward(data: Dict[str, Any]):
    """
    Internal endpoint for distributed forward pass.
    This would be used for more complex communication patterns.
    """
    if not distributed_inference:
        raise HTTPException(status_code=503, detail="Inference not initialized")
    
    # This is a placeholder for more complex distributed operations
    # In the current implementation, we use gRPC for direct communication
    raise HTTPException(status_code=501, detail="Not implemented")


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or device config
    port = int(os.environ.get("API_PORT", "8100"))
    
    uvicorn.run(app, host="0.0.0.0", port=port)