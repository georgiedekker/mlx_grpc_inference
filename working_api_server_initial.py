#!/usr/bin/env python3
"""
Working API server for distributed MLX inference.
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
import grpc

import sys
sys.path.insert(0, 'src')

from communication import inference_pb2_grpc, inference_pb2
from communication.dns_resolver import resolve_grpc_target
from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
from core.config import ClusterConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    model: str = "mlx-community/Qwen3-1.7B-8bit"
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
    
    logger.info("🚀 Starting MLX Distributed Inference API...")
    
    # Load configuration
    config = ClusterConfig.from_yaml('config/cluster_config.yaml')
    logger.info(f"✅ Config loaded: {config.model.name}")
    
    # Load model and tokenizer
    logger.info("📦 Loading model and tokenizer...")
    model, tokenizer = load(config.model.name)
    logger.info(f"✅ Model loaded: {len(model.layers) if hasattr(model, 'layers') else 'unknown'} layers")
    
    # Connect to workers
    logger.info("🔗 Connecting to workers...")
    workers = config.get_workers()
    
    for worker in workers:
        try:
            target = resolve_grpc_target(worker.hostname, worker.grpc_port)
            channel = grpc.insecure_channel(target)
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
    title="MLX Distributed Inference API",
    description="OpenAI-compatible API for distributed MLX inference", 
    version="1.0.0"
)

async def process_through_workers(hidden_states: mx.array) -> mx.array:
    """Process tensor through workers in rank order with parallel processing optimization."""
    sorted_workers = sorted(worker_connections.items(), key=lambda x: x[1]['rank'])
    logger.info(f"Processing through {len(sorted_workers)} workers")
    
    # Check if we can do any parallel processing (workers with non-overlapping layers)
    # For now, we need sequential processing since each worker depends on the previous
    # But we can optimize the individual calls
    
    for worker_id, worker_info in sorted_workers:
        stub = worker_info['stub']
        layers = worker_info['layers']
        
        # Serialize tensor without compression for compatibility
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
        logger.info(f"Sending tensor to {worker_id} for layers {layers} - shape: {hidden_states.shape}")
        import time
        start_time = time.time()
        layer_response = stub.ProcessLayers(layer_request, timeout=30.0)
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Received response from {worker_id} in {elapsed:.1f}ms")
        
        # Deserialize output
        output_metadata = {
            'shape': list(layer_response.metadata.shape),
            'dtype': layer_response.metadata.dtype,
            'compressed': layer_response.metadata.compressed
        }
        hidden_states = deserialize_mlx_array(layer_response.output_tensor, output_metadata)
    
    return hidden_states

async def process_through_workers_optimized_for_generation(hidden_states: mx.array, is_prefill: bool = False) -> mx.array:
    """Optimized worker processing for generation with reduced overhead."""
    sorted_workers = sorted(worker_connections.items(), key=lambda x: x[1]['rank'])
    
    # Use less logging during generation to reduce overhead
    if is_prefill:
        logger.info(f"Prefill: Processing through {len(sorted_workers)} workers")
    
    for worker_id, worker_info in sorted_workers:
        stub = worker_info['stub']
        layers = worker_info['layers']
        
        # Disable compression for now (workers expect gzip)
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
        
        # Process layers with reduced timeout for generation
        import time
        start_time = time.time()
        layer_response = stub.ProcessLayers(layer_request, timeout=10.0)  # Reduced timeout
        elapsed = (time.time() - start_time) * 1000
        
        if is_prefill:
            logger.info(f"Received response from {worker_id} in {elapsed:.1f}ms")
        
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
    return {"message": "MLX Distributed Inference API"}

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

@app.get("/cluster/status", response_model=ClusterStatus)
async def cluster_status():
    """Get cluster status."""
    workers_info = []
    for worker_id, worker_info in worker_connections.items():
        try:
            # Get device info
            device_info_response = worker_info['stub'].GetDeviceInfo(inference_pb2.Empty(), timeout=2.0)
            workers_info.append({
                "device_id": worker_id,
                "hostname": worker_info['hostname'],
                "status": "connected",
                "assigned_layers": list(device_info_response.assigned_layers),
                "gpu_utilization": device_info_response.gpu_utilization,
                "memory_usage_gb": device_info_response.memory_usage_gb
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
        "assigned_layers": config.model.get_device_layers(config.coordinator_device_id)
    }
    
    model_info = {
        "name": config.model.name,
        "total_layers": config.model.total_layers,
        "layer_distribution": {k: len(v) for k, v in config.model.layer_distribution.items()}
    }
    
    return ClusterStatus(
        coordinator=coordinator_info,
        workers=workers_info,
        model_info=model_info
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using distributed inference."""
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")
    
    try:
        overall_start_time = time.time()
        # Format messages into prompt
        prompt = format_messages(request.messages)
        
        # Tokenize input
        tokenize_start = time.time()
        input_ids = mx.array(tokenizer.encode(prompt))
        prompt_tokens = len(input_ids)
        tokenize_time = time.time() - tokenize_start
        
        # Track prompt processing time
        prompt_start_time = time.time()
        
        # Get embeddings - add batch dimension if needed
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # Add batch dimension if input_ids is 1D
            if len(input_ids.shape) == 1:
                input_ids = input_ids[None, :]  # Shape: [1, seq_len]
            embeddings = model.model.embed_tokens(input_ids)
        else:
            raise ValueError("Model does not have embed_tokens")
        
        # Process through coordinator's layers
        coordinator_layers = config.model.get_device_layers(config.coordinator_device_id)
        hidden_states = embeddings
        for layer_idx in coordinator_layers:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layer_output = model.model.layers[layer_idx](hidden_states)
                # Handle if layer returns tuple (hidden_states, attention_weights, ...)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
            else:
                layer_output = model.layers[layer_idx](hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
        
        # Process through workers (prefill phase)
        logger.info(f"Hidden states before workers - shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        hidden_states = await process_through_workers_optimized_for_generation(hidden_states, is_prefill=True)
        
        # Final norm and lm_head
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            hidden_states = model.model.norm(hidden_states)
        
        # Check various possible locations for lm_head
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(hidden_states)
        elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
            logits = model.model.lm_head(hidden_states)
        else:
            # For models with tied embeddings (like Qwen3), use embed_tokens.as_linear()
            logger.info("Model doesn't have lm_head, checking for tied embeddings...")
            
            # Check if model has tied embeddings
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                # Use the embedding layer's as_linear method for tied embeddings
                logits = model.model.embed_tokens.as_linear(hidden_states)
                logger.info(f"Used tied embeddings (embed_tokens.as_linear) - logits shape: {logits.shape}")
            else:
                logger.error(f"Model type: {type(model)}")
                logger.error(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                logger.error("Model does not have lm_head or embed_tokens for tied embeddings")
                raise ValueError("Model does not have accessible output projection")
        
        # Prompt processing complete
        prompt_end_time = time.time()
        prompt_processing_time = prompt_end_time - prompt_start_time
        prompt_tokens_per_second = prompt_tokens / prompt_processing_time if prompt_processing_time > 0 else 0
        
        # Generate tokens
        generation_start_time = time.time()
        sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
        generated_ids = []
        
        # Multi-token generation with optimizations
        current_ids = input_ids  # Start with original input
        max_new_tokens = min(request.max_tokens or 50, 50)  # Cap at 50 for better performance
        
        for token_idx in range(max_new_tokens):
            # Sample next token from current logits
            next_token = sampler(logits[:, -1:, :])
            token_id = next_token.item()
            generated_ids.append(token_id)
            
            # Check for stop tokens
            if token_id in [tokenizer.eos_token_id] or (request.stop and tokenizer.decode([token_id]) in request.stop):
                break
                
            # Append new token and continue generation
            new_token_tensor = mx.array([[token_id]])
            current_ids = mx.concatenate([current_ids, new_token_tensor], axis=1)
            
            # Re-run inference for next token (only through changed parts)
            # For efficiency, we only need to process the new token through embeddings and layers
            
            # Get embedding for new token
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                new_embedding = model.model.embed_tokens(new_token_tensor)
                # Concatenate with previous hidden states after embedding layer
                # For simplicity, we'll re-run the full forward pass for now
                # This can be optimized with KV-caching later
                
                # Re-embed all tokens (can be optimized with caching)
                embeddings = model.model.embed_tokens(current_ids)
                
                # Process through coordinator's layers
                hidden_states = embeddings
                for layer_idx in coordinator_layers:
                    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                        layer_output = model.model.layers[layer_idx](hidden_states)
                        if isinstance(layer_output, tuple):
                            hidden_states = layer_output[0]
                        else:
                            hidden_states = layer_output
                    else:
                        layer_output = model.layers[layer_idx](hidden_states)
                        if isinstance(layer_output, tuple):
                            hidden_states = layer_output[0]
                        else:
                            hidden_states = layer_output
                
                # Process through workers (generation phase)
                hidden_states = await process_through_workers_optimized_for_generation(hidden_states, is_prefill=False)
                
                # Final norm and get logits
                if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                    hidden_states = model.model.norm(hidden_states)
                
                # Get logits for next iteration
                if hasattr(model, 'lm_head'):
                    logits = model.lm_head(hidden_states)
                elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
                    logits = model.model.lm_head(hidden_states)
                else:
                    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                        logits = model.model.embed_tokens.as_linear(hidden_states)
                    else:
                        break  # Can't continue generation
        
        # Track generation time
        generation_end_time = time.time()
        generation_time = generation_end_time - generation_start_time
        completion_tokens = len(generated_ids)
        generation_tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
        
        # Decode
        generated_text = tokenizer.decode(generated_ids) if generated_ids else ""
        
        # Log performance metrics
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