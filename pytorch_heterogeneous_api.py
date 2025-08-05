#!/usr/bin/env python3
"""
FastAPI server for heterogeneous PyTorch distributed inference.
Provides OpenAI-compatible API endpoints.
"""
import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import argparse

import torch
import torch.distributed as dist
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.device_manager import DeviceManager
from pytorch_heterogeneous_server import HeterogeneousDistributedModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Heterogeneous Distributed Inference API")

# Global model instance
model: Optional[HeterogeneousDistributedModel] = None
device_manager: Optional[DeviceManager] = None


# Request/Response models
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict] = None
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "heterogeneous-cluster"


class ClusterStatus(BaseModel):
    status: str
    devices: List[Dict[str, Any]]
    model: Dict[str, Any]
    sharding_strategy: str
    layer_distribution: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize the distributed model on startup."""
    global model, device_manager
    
    try:
        # Get configuration
        model_name = os.getenv("MODEL_NAME", "microsoft/phi-2")
        config_file = os.getenv("CONFIG_FILE", "config/heterogeneous_cluster.json")
        
        logger.info(f"Initializing heterogeneous distributed model: {model_name}")
        logger.info(f"Using configuration: {config_file}")
        
        # Initialize device manager
        device_manager = DeviceManager(config_path=config_file)
        
        # Log device capabilities
        logger.info("Device capabilities:")
        for name, device in device_manager.devices.items():
            logger.info(f"  {name}: score={device.compute_score}, "
                       f"gpu={device.gpu_cores} cores, memory={device.gpu_memory_gb}GB")
        
        # Setup distributed environment for API server
        # API server runs as a separate process and doesn't participate in distributed
        
        # Create a simple interface to query the distributed workers
        # This would typically involve creating a client that communicates with the workers
        
        logger.info("Heterogeneous API server initialized")
        logger.info(f"Cluster using {len(device_manager.devices)} devices")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Heterogeneous Distributed Inference API",
        "endpoints": {
            "completions": "/v1/completions",
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "cluster/status": "/cluster/status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/cluster/status")
async def cluster_status():
    """Get detailed cluster status."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")
    
    # Get layer assignments
    from transformers import AutoConfig
    model_name = os.getenv("MODEL_NAME", "microsoft/phi-2")
    config = AutoConfig.from_pretrained(model_name)
    
    assignments = device_manager.get_layer_assignments(
        model_name=model_name,
        total_layers=config.num_hidden_layers
    )
    
    devices_info = []
    for name, device in device_manager.devices.items():
        assignment = assignments.get(name)
        device_info = {
            "name": name,
            "compute_score": device.compute_score,
            "gpu_cores": device.gpu_cores,
            "gpu_memory_gb": device.gpu_memory_gb,
            "status": "active",  # Would check actual worker status
            "layers": assignment.num_layers if assignment else 0,
            "layer_range": f"{assignment.start_layer}-{assignment.end_layer}" if assignment else "N/A"
        }
        devices_info.append(device_info)
    
    return ClusterStatus(
        status="operational",
        devices=devices_info,
        model={
            "name": model_name,
            "total_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads
        },
        sharding_strategy=device_manager.config.get('model', {}).get('sharding_strategy', 'capability_based'),
        layer_distribution={
            name: {
                "layers": assignment.layer_indices,
                "has_embeddings": assignment.has_embeddings,
                "has_lm_head": assignment.has_lm_head
            }
            for name, assignment in assignments.items()
        }
    )


@app.get("/v1/models")
async def list_models():
    """List available models."""
    model_name = os.getenv("MODEL_NAME", "microsoft/phi-2")
    
    return {
        "object": "list",
        "data": [
            ModelInfo(
                id=model_name,
                created=int(time.time())
            )
        ]
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a completion using the distributed model."""
    # For now, return a mock response
    # In production, this would communicate with the distributed workers
    
    model_name = os.getenv("MODEL_NAME", "microsoft/phi-2")
    
    # Generate a mock response
    completion_text = " Paris. The city is known for its iconic Eiffel Tower and rich cultural heritage."
    
    return CompletionResponse(
        id=f"cmpl-{int(time.time()*1000)}",
        created=int(time.time()),
        model=model_name,
        choices=[
            CompletionChoice(
                text=completion_text,
                index=0,
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(completion_text.split()),
            "total_tokens": len(request.prompt.split()) + len(completion_text.split())
        }
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using the distributed model."""
    # Convert chat format to prompt
    prompt = ""
    for message in request.messages:
        if message.role == "system":
            prompt += f"System: {message.content}\n"
        elif message.role == "user":
            prompt += f"User: {message.content}\n"
        elif message.role == "assistant":
            prompt += f"Assistant: {message.content}\n"
    prompt += "Assistant:"
    
    # Use completion endpoint logic
    completion_request = CompletionRequest(
        model=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        stream=request.stream,
        stop=request.stop
    )
    
    # Get completion (mock for now)
    response_text = " I'm a helpful AI assistant. How can I help you today?"
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time()*1000)}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split())
        }
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Heterogeneous Distributed Inference API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()
    
    # Run the server
    uvicorn.run(
        "pytorch_heterogeneous_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()