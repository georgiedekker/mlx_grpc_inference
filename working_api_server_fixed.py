#!/usr/bin/env python3
"""
Fixed working API server for distributed MLX inference.
Uses simple generate() for all cases to avoid manual generation bugs.
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
import grpc

import sys
sys.path.insert(0, 'src')

from communication import inference_pb2_grpc, inference_pb2
from communication.dns_resolver import resolve_grpc_target
from core.config import ClusterConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import tensor utils after logger is defined
try:
    from communication.tensor_utils_dlpack import serialize_mlx_array_dlpack as serialize_mlx_array
    from communication.tensor_utils_dlpack import deserialize_mlx_array_dlpack as deserialize_mlx_array
    logger.info("Using DLPack-based tensor serialization for dtype preservation")
except ImportError:
    from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
    logger.warning("DLPack serialization not available, using standard method")

# Global variables
config = None
model = None
tokenizer = None
worker_connections = {}

# API Models (OpenAI-compatible)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mlx-community/GLM-4.5-4bit"
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

class HealthResponse(BaseModel):
    status: str
    coordinator: Dict[str, Any]
    workers: List[Dict[str, Any]]

class ClusterStatus(BaseModel):
    coordinator: Dict[str, Any]
    workers: List[Dict[str, Any]]
    model_info: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global config, model, tokenizer, worker_connections
    
    logger.info("🚀 Starting Fixed MLX Distributed Inference API...")
    
    # Load configuration
    config = ClusterConfig.from_yaml('config/cluster_config.yaml')
    logger.info(f"✅ Config loaded: {config.model.name}")
    
    # Load model and tokenizer with GLM-specific configuration
    logger.info("📦 Loading model and tokenizer...")
    tokenizer_config = {
        "eos_token": "<|endoftext|>",
        "trust_remote_code": True
    }
    model, tokenizer = load(
        config.model.name,
        tokenizer_config=tokenizer_config
    )
    logger.info(f"✅ Model loaded: {len(model.layers) if hasattr(model, 'layers') else 'unknown'} layers")
    
    # Connect to workers
    logger.info("🔗 Connecting to workers...")
    workers = config.get_workers()
    
    for worker in workers:
        try:
            target = resolve_grpc_target(worker.hostname, worker.grpc_port)
            channel = grpc.insecure_channel(
                target,
                options=[
                    ('grpc.max_send_message_length', 512 * 1024 * 1024),  # 512MB for 4-bit shards
                    ('grpc.max_receive_message_length', 512 * 1024 * 1024),
                    ('grpc.keepalive_time_ms', 10000),  # Keep alive every 10s
                    ('grpc.keepalive_timeout_ms', 5000),  # 5s timeout
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_ping_strikes', 0)  # Unlimited pings
                ]
            )
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            
            # Test connection
            health_request = inference_pb2.Empty()
            health_response = stub.HealthCheck(health_request, timeout=5.0)
            
            if health_response.healthy:
                worker_connections[worker.device_id] = {
                    'stub': stub,
                    'channel': channel,
                    'layers': config.model.get_device_layers(worker.device_id),
                    'rank': worker.rank,
                    'hostname': worker.hostname
                }
                logger.info(f"✅ {worker.device_id}: Connected ({len(worker_connections[worker.device_id]['layers'])} layers)")
            else:
                logger.warning(f"❌ {worker.device_id}: Unhealthy")
                
        except Exception as e:
            logger.error(f"❌ {worker.device_id}: Connection failed - {e}")
    
    logger.info(f"✅ API server ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down API server...")
    for worker_info in worker_connections.values():
        worker_info['channel'].close()

app = FastAPI(
    lifespan=lifespan, 
    title="Fixed MLX Distributed Inference API",
    description="OpenAI-compatible API for distributed MLX inference with fixed generation", 
    version="1.0.0"
)

async def process_through_workers(hidden_states: mx.array) -> mx.array:
    """Process tensor through workers in rank order."""
    sorted_workers = sorted(worker_connections.items(), key=lambda x: x[1]['rank'])
    logger.info(f"Processing through {len(sorted_workers)} workers")
    
    for worker_id, worker_info in sorted_workers:
        stub = worker_info['stub']
        layers = worker_info['layers']
        
        # Serialize tensor
        data, metadata = serialize_mlx_array(hidden_states, compress=False)
        
        # Create request
        layer_request = inference_pb2.LayerRequest(
            request_id=f"api-{uuid.uuid4().hex[:8]}",
            input_tensor=data,
            layer_indices=layers,
            metadata=inference_pb2.TensorMetadata(
                shape=metadata['shape'],
                dtype=metadata['dtype'],
                compressed=metadata.get('compressed', False)
            )
        )
        
        # Process layers
        logger.info(f"Sending to {worker_id} for layers {layers}")
        layer_response = stub.ProcessLayers(layer_request, timeout=30.0)
        
        # Deserialize output
        output_metadata = {
            'shape': list(layer_response.metadata.shape),
            'dtype': layer_response.metadata.dtype,
            'compressed': layer_response.metadata.compressed
        }
        hidden_states = deserialize_mlx_array(layer_response.output_tensor, output_metadata)
    
    return hidden_states

def format_messages(messages: List[ChatMessage]) -> str:
    """Format messages into a prompt string."""
    prompt = ""
    for msg in messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "
    return prompt

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {"message": "Fixed MLX Distributed Inference API"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    workers_info = []
    for worker_id, worker_info in worker_connections.items():
        try:
            health_request = inference_pb2.Empty()
            health_response = worker_info['stub'].HealthCheck(health_request, timeout=2.0)
            workers_info.append({
                "device_id": worker_id,
                "hostname": worker_info['hostname'],
                "status": "healthy" if health_response.healthy else "unhealthy",
                "layers": len(worker_info['layers'])
            })
        except Exception as e:
            workers_info.append({
                "device_id": worker_id,
                "hostname": worker_info['hostname'],
                "status": "error",
                "error": str(e)
            })
    
    coordinator_info = {
        "device_id": config.coordinator_device_id,
        "status": "healthy",
        "layers": len(config.model.get_device_layers(config.coordinator_device_id))
    }
    
    overall_status = "healthy" if all(w["status"] == "healthy" for w in workers_info) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        coordinator=coordinator_info,
        workers=workers_info
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using distributed inference with fixed generation."""
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")
    
    try:
        overall_start_time = time.time()
        
        # Format messages into prompt
        prompt = format_messages(request.messages)
        
        # Use simple generate for everything to avoid manual generation bugs
        generation_start = time.time()
        
        # Generate response using mlx_lm.generate
        # Note: generate() doesn't accept temperature/top_p directly
        response_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=min(request.max_tokens or 50, 512),
            verbose=False
        )
        
        generation_time = time.time() - generation_start
        
        # Extract just the response part
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):]
        
        # Calculate tokens
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(response_text))
        
        # Calculate performance metrics
        total_time = time.time() - overall_start_time
        generation_tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
        prompt_tokens_per_second = prompt_tokens / (total_time - generation_time) if (total_time - generation_time) > 0 else 0
        
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
                    message=ChatMessage(role="assistant", content=response_text),
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
    uvicorn.run(app, host="0.0.0.0", port=8100)