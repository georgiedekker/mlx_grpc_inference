"""
OpenAI-compatible API server for distributed inference.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..core.config import ClusterConfig, DeviceRole
from ..model.loader import DistributedModelLoader
from ..model.inference import LayerProcessor
from ..communication.grpc_server import create_grpc_server
from .orchestrator import DistributedOrchestrator
from .request_handler import InferenceRequest

logger = logging.getLogger(__name__)


# OpenAI-compatible API models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mlx-community/Qwen3-1.7B-8bit"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = 512
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device_id: str
    role: str
    workers_connected: int
    timestamp: int


class ClusterStatus(BaseModel):
    coordinator: Dict[str, Any]
    workers: List[Dict[str, Any]]
    model_info: Dict[str, Any]


# Global state
config: Optional[ClusterConfig] = None
orchestrator: Optional[DistributedOrchestrator] = None
grpc_server = None
model_loader: Optional[DistributedModelLoader] = None
layer_processor: Optional[LayerProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()


# Create FastAPI app
app = FastAPI(
    title="MLX Distributed Inference API",
    description="OpenAI-compatible API for distributed MLX inference",
    version="1.0.0",
    lifespan=lifespan
)


async def startup_event():
    """Initialize the API server."""
    global config, orchestrator, grpc_server, model_loader, layer_processor
    
    try:
        # Load configuration
        config = ClusterConfig.from_yaml("config/cluster_config.yaml")
        device_config = config.get_local_device()
        
        if not device_config:
            raise ValueError("Could not determine local device configuration")
        
        logger.info(f"Starting API server on {device_config.device_id}")
        
        if device_config.role == DeviceRole.COORDINATOR:
            # Initialize orchestrator for coordinator
            orchestrator = DistributedOrchestrator(config)
            await orchestrator.initialize()
            
            # Also start gRPC server for coordinator's layers
            model_loader = DistributedModelLoader(config)
            model, _ = model_loader.load_model_shard(device_config.device_id)
            
            if model:
                assigned_layers = config.model.get_device_layers(device_config.device_id)
                layer_processor = LayerProcessor(model, device_config.device_id, assigned_layers)
                
                grpc_server = create_grpc_server(config, device_config, layer_processor)
                grpc_server.start()
                logger.info(f"gRPC server started on port {device_config.grpc_port}")
        else:
            # This is a worker - it shouldn't run the API server
            logger.warning(f"Device {device_config.device_id} is a worker, not starting API server")
        
        logger.info("API server initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API server: {e}")
        raise


async def shutdown_event():
    """Cleanup on shutdown."""
    global grpc_server
    
    if grpc_server:
        grpc_server.stop(grace_period=10)
        logger.info("gRPC server stopped")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "MLX Distributed Inference API",
        "status": "running",
        "device_id": config.get_local_device().device_id if config else "unknown"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device = config.get_local_device() if config else None
    
    workers_connected = 0
    if orchestrator and orchestrator.connection_pool:
        workers_connected = len(orchestrator.connection_pool.clients)
    
    return HealthResponse(
        status="healthy" if orchestrator and orchestrator.is_initialized else "initializing",
        model_loaded=orchestrator is not None and orchestrator.model is not None,
        device_id=device.device_id if device else "unknown",
        role=device.role.value if device else "unknown",
        workers_connected=workers_connected,
        timestamp=int(time.time())
    )


@app.get("/cluster/status", response_model=ClusterStatus)
async def cluster_status():
    """Get cluster status."""
    if not config or not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    coordinator_info = {
        "device_id": config.coordinator_device_id,
        "status": "active",
        "model_loaded": orchestrator.model is not None
    }
    
    workers_info = []
    for worker in config.get_workers():
        client = orchestrator.connection_pool.get_client(worker.device_id) if orchestrator.connection_pool else None
        
        if client:
            try:
                info = client.get_device_info()
                workers_info.append({
                    "device_id": worker.device_id,
                    "status": "connected",
                    "assigned_layers": info['assigned_layers'],
                    "gpu_utilization": info['gpu_utilization'],
                    "memory_usage_gb": info['memory_usage_gb']
                })
            except:
                workers_info.append({
                    "device_id": worker.device_id,
                    "status": "error"
                })
        else:
            workers_info.append({
                "device_id": worker.device_id,
                "status": "disconnected"
            })
    
    model_info = {}
    if model_loader and orchestrator and orchestrator.model:
        model_info = model_loader.get_model_info(orchestrator.model)
    
    return ClusterStatus(
        coordinator=coordinator_info,
        workers=workers_info,
        model_info=model_info
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    if not orchestrator or not orchestrator.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Check if streaming is requested
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")
    
    try:
        # Create inference request
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        inference_request = InferenceRequest(
            request_id=request_id,
            messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=1.1,  # Convert from frequency_penalty
            stream=request.stream,
            stop_sequences=request.stop
        )
        
        # Process request
        logger.info(f"Processing request {request_id}")
        response = await orchestrator.process_request(inference_request)
        
        # Format response
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response.content),
                    finish_reason=response.stop_reason if hasattr(response, 'stop_reason') else "stop"
                )
            ],
            usage=Usage(
                prompt_tokens=100,  # Estimated
                completion_tokens=response.tokens_generated,
                total_tokens=100 + response.tokens_generated
            )
        )
        
    except Exception as e:
        logger.error(f"Error processing chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models."""
    if not config:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "object": "list",
        "data": [
            {
                "id": config.model.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-community"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="MLX Distributed Inference API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "src.coordinator.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )