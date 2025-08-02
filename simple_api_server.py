#!/usr/bin/env python3
"""
Simple API server that bypasses complex initialization and directly uses working components.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load
import grpc

import sys
sys.path.insert(0, 'src')

from communication import inference_pb2_grpc, inference_pb2
from communication.dns_resolver import resolve_grpc_target
from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
from core.config import ClusterConfig

# Global variables
config = None
model = None
tokenizer = None
worker_stubs = {}

# API Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mlx-community/Qwen3-1.7B-8bit"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 512

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    global config, model, tokenizer, worker_stubs
    
    print("🚀 Starting Simple MLX API Server...")
    
    # Load configuration
    config = ClusterConfig.from_yaml('config/cluster_config.yaml')
    print(f"✅ Config loaded: {config.model.name}")
    
    # Load model and tokenizer
    print("📦 Loading model and tokenizer...")
    model, tokenizer = load(config.model.name)
    print(f"✅ Model loaded: {len(model.layers)} layers")
    
    # Connect to workers
    print("🔗 Connecting to workers...")
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
                worker_stubs[worker.device_id] = {
                    'stub': stub,
                    'channel': channel,
                    'layers': config.model.get_device_layers(worker.device_id),
                    'rank': worker.rank
                }
                print(f"✅ {worker.device_id}: Connected, {len(worker_stubs[worker.device_id]['layers'])} layers")
            else:
                print(f"❌ {worker.device_id}: Unhealthy")
                
        except Exception as e:
            print(f"❌ {worker.device_id}: Connection failed - {e}")
    
    print(f"✅ API server ready at http://0.0.0.0:8100")
    
    yield
    
    # Shutdown
    print("👋 Shutting down API server...")
    for worker_info in worker_stubs.values():
        worker_info['channel'].close()

app = FastAPI(lifespan=lifespan, title="MLX Distributed Inference API")

async def distributed_forward(input_ids: mx.array) -> mx.array:
    """Run distributed forward pass through workers."""
    # Get embeddings from model
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embeddings = model.model.embed_tokens(input_ids)
    else:
        raise ValueError("Model does not have embed_tokens")
    
    # Process through coordinator's layers (0-9)
    coordinator_layers = config.model.get_device_layers(config.coordinator_device_id)
    hidden_states = embeddings
    for layer_idx in coordinator_layers:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            hidden_states = model.model.layers[layer_idx](hidden_states)
        else:
            hidden_states = model.layers[layer_idx](hidden_states)
    
    # Process through workers in order
    sorted_workers = sorted(worker_stubs.items(), key=lambda x: x[1]['rank'])
    
    for worker_id, worker_info in sorted_workers:
        stub = worker_info['stub']
        layers = worker_info['layers']
        
        # Serialize tensor
        data, metadata = serialize_mlx_array(hidden_states)
        
        # Create request
        layer_request = inference_pb2.LayerRequest(
            request_id=f"api-{uuid.uuid4().hex}",
            input_tensor=data,
            layer_indices=layers,
            metadata=inference_pb2.TensorMetadata(
                shape=metadata['shape'],
                dtype=metadata['dtype'],
                compressed=metadata.get('compressed', False)
            )
        )
        
        # Process layers
        layer_response = stub.ProcessLayers(layer_request, timeout=30.0)
        
        # Deserialize output
        output_metadata = {
            'shape': list(layer_response.metadata.shape),
            'dtype': layer_response.metadata.dtype,
            'compressed': layer_response.metadata.compressed
        }
        hidden_states = deserialize_mlx_array(layer_response.output_tensor, output_metadata)
    
    return hidden_states

async def generate_response(prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate text using distributed inference."""
    # Tokenize input
    input_ids = mx.array(tokenizer.encode(prompt))
    
    # Get initial hidden states through distributed pipeline
    hidden_states = await distributed_forward(input_ids[None, :])  # Add batch dimension
    
    # Process through output layers
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        hidden_states = model.model.norm(hidden_states)
    
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(hidden_states)
    else:
        raise ValueError("Model does not have lm_head")
    
    # Sample from logits
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temperature)
    
    generated_ids = []
    
    for _ in range(max_tokens):
        # Sample next token
        next_token = sampler(logits[:, -1:, :])
        token_id = next_token.item()
        generated_ids.append(token_id)
        
        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            break
        
        # For simplicity, just break after first token for now
        # Full implementation would run new token through pipeline
        break
    
    # Decode generated tokens
    if generated_ids:
        return tokenizer.decode(generated_ids)
    return ""

@app.get("/health")
async def health():
    """Health check endpoint."""
    worker_health = {}
    for worker_id, worker_info in worker_stubs.items():
        try:
            health_request = inference_pb2.Empty()
            health_response = worker_info['stub'].HealthCheck(health_request, timeout=2.0)
            worker_health[worker_id] = health_response.healthy
        except:
            worker_health[worker_id] = False
    
    all_healthy = all(worker_health.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "workers": worker_health,
        "model": config.model.name if config else "not loaded"
    }

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Format messages into prompt
        prompt = ""
        for msg in request.messages:
            prompt += f"{msg.role}: {msg.content}\n"
        prompt += "assistant: "
        
        # Generate response
        start_time = time.time()
        generated_text = await generate_response(prompt, request.max_tokens, request.temperature)
        generation_time = time.time() - start_time
        
        # Count tokens
        input_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(generated_text))
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)