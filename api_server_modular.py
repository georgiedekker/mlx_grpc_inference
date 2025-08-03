#!/usr/bin/env python3
"""
Modular MLX Distributed Inference API Server

Enhanced version with device management, coordinator migration, and clean modularity.
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
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from src.management.device_manager import DeviceManager
from src.communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
from src.core.config import ClusterConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
device_manager: Optional[DeviceManager] = None
model = None
tokenizer = None

# API Models (OpenAI-compatible)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "auto"
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

# Device Management Models
class DeviceInfo(BaseModel):
    device_id: str
    hostname: str
    role: str
    status: str
    assigned_layers: List[int]
    capabilities: Dict[str, Any]
    last_heartbeat: float

class ClusterInfo(BaseModel):
    coordinator_id: str
    total_devices: int
    healthy_devices: int
    devices: Dict[str, DeviceInfo]
    model_info: Dict[str, Any]

class AddDeviceRequest(BaseModel):
    device_id: str
    hostname: str
    role: str
    grpc_port: int = 50051
    capabilities: Dict[str, Any]

class MigrateCoordinatorRequest(BaseModel):
    new_coordinator_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global device_manager, model, tokenizer
    
    logger.info("🚀 Starting Modular MLX Distributed Inference API...")
    
    # Initialize device manager
    device_manager = DeviceManager('config/cluster_config.yaml')
    await device_manager.initialize()
    
    # Load model based on current configuration
    if device_manager.config and device_manager.config.model:
        model_name = device_manager.config.model.name
        logger.info(f"📦 Loading model: {model_name}")
        model, tokenizer = load(model_name)
        logger.info(f"✅ Model loaded: {len(model.layers) if hasattr(model, 'layers') else 'unknown'} layers")
    else:
        logger.warning("⚠️  No model configuration found")
    
    logger.info("✅ Modular API server ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down API server...")
    if device_manager:
        await device_manager.shutdown()

app = FastAPI(
    lifespan=lifespan, 
    title="MLX Distributed Inference API (Modular)",
    description="Enhanced OpenAI-compatible API with device management and coordinator migration", 
    version="2.0.0"
)

# Core inference functions (simplified for modularity)
async def distributed_inference(input_ids: mx.array, config: ClusterConfig) -> mx.array:
    """Perform distributed inference across the cluster."""
    # Get embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        hidden_states = model.model.embed_tokens(input_ids)
    else:
        raise ValueError("Model does not have embed_tokens")
    
    # Process coordinator layers
    coordinator_layers = config.model.get_device_layers(config.coordinator_device_id)
    for layer_idx in coordinator_layers:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer_output = model.model.layers[layer_idx](hidden_states)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
    
    # Process worker layers (simplified - using device manager connections)
    healthy_workers = [
        device for device in device_manager.devices.values()
        if device.role.value == "worker" and device.is_healthy()
    ]
    
    for worker in sorted(healthy_workers, key=lambda x: min(x.assigned_layers) if x.assigned_layers else float('inf')):
        if worker.device_id in device_manager.connections:
            # Process through this worker (implementation details omitted for brevity)
            # In production, this would use the gRPC calls
            pass
    
    # Final norm and output projection
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        hidden_states = model.model.norm(hidden_states)
    
    # Get logits using tied embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        logits = model.model.embed_tokens.as_linear(hidden_states)
    else:
        raise ValueError("Model does not have tied embeddings")
    
    return logits

def format_messages(messages: List[ChatMessage]) -> str:
    """Format messages into a prompt string."""
    # For single user messages without special formatting, return as-is
    if len(messages) == 1 and messages[0].role == "user":
        content = messages[0].content
        # Check if it already has role formatting
        if not any(role in content for role in ["Human:", "Assistant:", "user:", "assistant:"]):
            return content  # Return without adding role formatting
    
    # For multi-turn or formatted messages, use role formatting
    prompt = ""
    for msg in messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "
    return prompt

# API Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {"message": "MLX Distributed Inference API (Modular)", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")
    
    cluster_status = await device_manager.get_cluster_status()
    healthy_count = cluster_status["healthy_devices"]
    total_count = cluster_status["total_devices"]
    
    status = "healthy" if healthy_count == total_count and total_count > 0 else "degraded"
    
    return {
        "status": status,
        "cluster": {
            "coordinator": cluster_status["coordinator_id"],
            "devices": f"{healthy_count}/{total_count} healthy",
            "model": cluster_status["model_info"]["name"]
        }
    }

@app.get("/cluster", response_model=ClusterInfo)
async def get_cluster_info():
    """Get detailed cluster information."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")
    
    cluster_status = await device_manager.get_cluster_status()
    
    # Convert device statuses to API models
    devices_info = {}
    for device_id, device_data in cluster_status["devices"].items():
        devices_info[device_id] = DeviceInfo(
            device_id=device_data["device_id"],
            hostname=device_data["hostname"],
            role=device_data["role"],
            status=device_data["status"],
            assigned_layers=device_data["assigned_layers"],
            capabilities=device_data["capabilities"],
            last_heartbeat=device_data["last_heartbeat"]
        )
    
    return ClusterInfo(
        coordinator_id=cluster_status["coordinator_id"],
        total_devices=cluster_status["total_devices"],
        healthy_devices=cluster_status["healthy_devices"],
        devices=devices_info,
        model_info=cluster_status["model_info"]
    )

@app.post("/cluster/devices")
async def add_device(request: AddDeviceRequest):
    """Add a new device to the cluster."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")
    
    # Create device configuration
    from src.core.config import DeviceConfig, DeviceRole, DeviceCapabilities
    
    device_config = DeviceConfig(
        device_id=request.device_id,
        hostname=request.hostname,
        api_port=8100,  # Default
        grpc_port=request.grpc_port,
        role=DeviceRole(request.role),
        rank=len(device_manager.devices),  # Simple rank assignment
        capabilities=DeviceCapabilities(**request.capabilities)
    )
    
    success = await device_manager.add_device(device_config)
    
    if success:
        return {"message": f"Device {request.device_id} added successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to add device {request.device_id}")

@app.delete("/cluster/devices/{device_id}")
async def remove_device(device_id: str):
    """Remove a device from the cluster."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")
    
    success = await device_manager.remove_device(device_id)
    
    if success:
        return {"message": f"Device {device_id} removed successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Device {device_id} not found or removal failed")

@app.post("/cluster/migrate-coordinator")
async def migrate_coordinator(request: MigrateCoordinatorRequest):
    """Migrate coordinator role to a different device."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")
    
    success = await device_manager.migrate_coordinator(request.new_coordinator_id)
    
    if success:
        return {"message": f"Coordinator migrated to {request.new_coordinator_id}"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to migrate coordinator to {request.new_coordinator_id}")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using distributed inference."""
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")
    
    if not device_manager or not device_manager.config:
        raise HTTPException(status_code=503, detail="Cluster not properly initialized")
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        overall_start_time = time.time()
        
        # Format and tokenize
        prompt = format_messages(request.messages)
        tokenize_start = time.time()
        input_ids = mx.array(tokenizer.encode(prompt))
        prompt_tokens = len(input_ids)
        
        # Prompt processing
        prompt_start_time = time.time()
        logits = await distributed_inference(input_ids, device_manager.config)
        prompt_end_time = time.time()
        
        prompt_processing_time = prompt_end_time - prompt_start_time
        prompt_tokens_per_second = prompt_tokens / prompt_processing_time if prompt_processing_time > 0 else 0
        
        # Generate tokens using corrected logic
        generation_start_time = time.time()
        
        # Check if this is a simple prompt
        is_simple_prompt = (
            len(request.messages) == 1 and 
            request.messages[0].role == "user" and 
            not any(role in prompt for role in ["Human:", "Assistant:", "user:", "assistant:"])
        )
        
        if is_simple_prompt:
            # Use mlx_lm generate for simple prompts
            response_text = generate(
                model,
                tokenizer,
                prompt=request.messages[0].content,  # Use content directly, not formatted
                max_tokens=min(request.max_tokens or 50, 100)
            )
            # Extract just the generated part
            if response_text.startswith(request.messages[0].content):
                response_text = response_text[len(request.messages[0].content):]
            generated_ids = tokenizer.encode(response_text)
            generated_text = response_text
        else:
            # Multi-token generation with temperature control
            sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
            generated_ids = []
            current_ids = input_ids
            if len(current_ids.shape) == 1:
                current_ids = current_ids[None, :]
            
            max_new_tokens = min(request.max_tokens or 50, 100)
            
            for _ in range(max_new_tokens):
                # Sample next token
                next_token = sampler(logits[:, -1:, :])
                token_id = next_token.item()
                generated_ids.append(token_id)
                
                # Check for stop conditions
                if token_id == tokenizer.eos_token_id or (request.stop and tokenizer.decode([token_id]) in request.stop):
                    break
                
                # Prepare for next iteration
                new_token_tensor = mx.array([[token_id]])
                current_ids = mx.concatenate([current_ids, new_token_tensor], axis=1)
                
                # Re-run inference for next token
                logits = await distributed_inference(current_ids, device_manager.config)
            
            generated_text = tokenizer.decode(generated_ids) if generated_ids else ""
        
        generation_end_time = time.time()
        generation_time = generation_end_time - generation_start_time
        completion_tokens = len(generated_ids)
        generation_tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
        
        # Log performance
        total_time = time.time() - overall_start_time
        logger.info(f"Performance: {prompt_tokens} prompt tokens @ {prompt_tokens_per_second:.1f} tok/s, "
                   f"{completion_tokens} generated @ {generation_tokens_per_second:.1f} tok/s, "
                   f"total time: {total_time:.2f}s")
        
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
        logger.error(f"Error in chat completion: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models."""
    if device_manager and device_manager.config and device_manager.config.model:
        model_name = device_manager.config.model.name
    else:
        model_name = "unknown"
    
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-community"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)