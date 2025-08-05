#!/usr/bin/env python3
"""
Hybrid distributed server - gRPC orchestration with MLX optimized tensor ops.
Uses Thunderbolt network for communication between devices.
"""
import os
import asyncio
import grpc
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import mlx.core as mx
from mlx_lm import load, generate
from typing import List, Dict, Any
from concurrent import futures

from src.communication import inference_pb2, inference_pb2_grpc
from src.communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array

# Wrapper functions to match proto format
def serialize_tensor(array: mx.array) -> inference_pb2.Tensor:
    """Serialize MLX array to proto Tensor."""
    data, metadata = serialize_mlx_array(array, compress=True, compression_algorithm="lz4")
    tensor = inference_pb2.Tensor()
    tensor.data = data
    tensor.shape.extend(metadata['shape'])
    tensor.dtype = metadata['dtype']
    return tensor

def deserialize_tensor(tensor: inference_pb2.Tensor) -> mx.array:
    """Deserialize proto Tensor to MLX array."""
    metadata = {
        'shape': list(tensor.shape),
        'dtype': tensor.dtype,
        'compressed': True,
        'compression_info': {'algorithm': 'lz4'}
    }
    return deserialize_mlx_array(tensor.data, metadata)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="MLX Distributed Inference")

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# Global state
model = None
tokenizer = None
worker_stubs = []
model_config = None

def initialize_coordinator():
    """Initialize the coordinator with model metadata and worker connections."""
    global model, tokenizer, worker_stubs, model_config
    
    # Load model to get configuration
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    logger.info(f"Loading model configuration from {model_name}")
    model, tokenizer = load(model_name, lazy=True)
    
    # Get model configuration
    model_config = {
        "n_layers": len(model.model.layers),
        "vocab_size": model.model.vocab_size,
        "model_type": model.model_type,
    }
    logger.info(f"Model has {model_config['n_layers']} layers")
    
    # Connect to workers on Thunderbolt network
    worker_addresses = [
        "192.168.5.1:50051",  # mini1 (coordinator also acts as worker)
        "192.168.5.2:50051",  # mini2 
        "192.168.5.3:50051",  # m4 master
    ]
    
    # Distribute layers across workers
    layers_per_worker = model_config['n_layers'] // len(worker_addresses)
    remainder = model_config['n_layers'] % len(worker_addresses)
    
    layer_assignments = []
    start_layer = 0
    
    logger.info(f"Distributing {model_config['n_layers']} layers across {len(worker_addresses)} workers")
    logger.info(f"Base layers per worker: {layers_per_worker}, remainder: {remainder}")
    
    for i, addr in enumerate(worker_addresses):
        # Give extra layers to first workers if there's a remainder
        num_layers = layers_per_worker + (1 if i < remainder else 0)
        end_layer = start_layer + num_layers
        
        layer_assignments.append({
            "address": addr,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "is_first": i == 0,
            "is_last": i == len(worker_addresses) - 1
        })
        
        logger.info(f"Worker {i} ({addr}): layers {start_layer}-{end_layer-1} ({num_layers} layers)")
        start_layer = end_layer
    
    # Connect to workers
    for assignment in layer_assignments:
        addr = assignment["address"]
        logger.info(f"Connecting to worker at {addr} (layers {assignment['start_layer']}-{assignment['end_layer']-1})")
        
        if addr.startswith("192.168.5.1"):
            # Local worker - we'll handle this directly
            worker_stubs.append({
                "stub": None,  # Local processing
                "assignment": assignment
            })
        else:
            # Remote worker
            channel = grpc.insecure_channel(addr, options=[
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024),
            ])
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            worker_stubs.append({
                "stub": stub,
                "assignment": assignment
            })
    
    logger.info("Coordinator initialized successfully")

async def distributed_forward(input_ids: mx.array) -> mx.array:
    """Forward pass through distributed model."""
    current_hidden = None
    total_start = time.time()
    
    # Get embeddings (on first worker)
    if worker_stubs[0]["stub"] is None:
        # Local processing
        with mx.stream(mx.gpu):
            current_hidden = model.model.embed_tokens(input_ids)
            mx.eval(current_hidden)
    else:
        # Remote processing
        request = inference_pb2.ForwardRequest()
        request.input_ids.extend(input_ids.tolist()[0])
        request.is_embedding = True
        
        response = worker_stubs[0]["stub"].Forward(request)
        current_hidden = deserialize_tensor(response.output)
    
    # Process through transformer layers
    for worker_info in worker_stubs:
        assignment = worker_info["assignment"]
        stub = worker_info["stub"]
        
        if stub is None:
            # Local processing
            layer_start = time.time()
            logger.info(f"Processing layers {assignment['start_layer']}-{assignment['end_layer']-1} locally")
            with mx.stream(mx.gpu):
                for i in range(assignment["start_layer"], assignment["end_layer"]):
                    current_hidden = model.model.layers[i](current_hidden)
                mx.eval(current_hidden)
            logger.info(f"Local layers took {time.time() - layer_start:.3f}s")
        else:
            # Remote processing
            layer_start = time.time()
            logger.info(f"Sending to worker at {assignment['address']} for layers {assignment['start_layer']}-{assignment['end_layer']-1}")
            request = inference_pb2.LayerRequestV2()
            request.input_tensor.CopyFrom(serialize_tensor(current_hidden))
            request.start_layer = assignment["start_layer"]
            request.end_layer = assignment["end_layer"]
            
            send_time = time.time()
            response = stub.ProcessLayers(request)
            recv_time = time.time()
            current_hidden = deserialize_tensor(response.output_tensor)
            logger.info(f"Remote layers took {recv_time - layer_start:.3f}s (send: {send_time - layer_start:.3f}s, process+recv: {recv_time - send_time:.3f}s)")
    
    # Final projection (on last worker)
    last_worker = worker_stubs[-1]
    if last_worker["stub"] is None:
        # Local processing
        with mx.stream(mx.gpu):
            current_hidden = model.model.norm(current_hidden)
            # Most MLX models use tied embeddings
            if hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'as_linear'):
                # Use tied embeddings (most common case)
                logits = model.model.embed_tokens.as_linear(current_hidden)
            elif hasattr(model, 'lm_head') and model.lm_head is not None:
                # Use separate lm_head if available
                logits = model.lm_head(current_hidden)
            else:
                logger.error(f"Cannot find output projection. Model type: {type(model)}")
                raise AttributeError("Cannot find output projection layer")
            mx.eval(logits)
    else:
        # Remote processing
        request = inference_pb2.ForwardRequest()
        request.input_tensor.CopyFrom(serialize_tensor(current_hidden))
        request.is_final_projection = True
        
        response = last_worker["stub"].Forward(request)
        logits = deserialize_tensor(response.output)
    
    logger.info(f"Total forward pass took {time.time() - total_start:.3f}s")
    return logits

async def generate_distributed(prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate text using distributed model."""
    # Tokenize
    input_ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    
    generated_tokens = []
    
    for _ in range(max_tokens):
        # Forward pass
        logits = await distributed_forward(input_ids)
        
        # Sample next token
        with mx.stream(mx.gpu):
            # Get logits for last position
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply softmax and sample
            probs = mx.softmax(next_token_logits)
            next_token = mx.random.categorical(mx.log(probs))
            mx.eval(next_token)
        
        generated_tokens.append(int(next_token.item()))
        
        # Update input_ids
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)
        
        # Stop if EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode
    full_tokens = tokenizer.encode(prompt) + generated_tokens
    return tokenizer.decode(full_tokens)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Format messages into prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        prompt += "Assistant: "
        
        # Generate response
        start_time = time.time()
        response_text = await generate_distributed(prompt, request.max_tokens, request.temperature)
        elapsed = time.time() - start_time
        
        # Extract just the assistant's response
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):]
        
        # Count tokens
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(response_text))
        
        logger.info(f"Generated {completion_tokens} tokens in {elapsed:.2f}s ({completion_tokens/elapsed:.1f} tok/s)")
        
        # Return OpenAI-compatible response
        return ChatResponse(
            id=f"chatcmpl-{os.urandom(8).hex()}",
            created=int(time.time()),
            model="mlx-community/Qwen3-1.7B-8bit",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    worker_status = []
    for i, worker_info in enumerate(worker_stubs):
        if worker_info["stub"] is None:
            worker_status.append({"worker": i, "status": "local", "layers": f"{worker_info['assignment']['start_layer']}-{worker_info['assignment']['end_layer']-1}"})
        else:
            try:
                response = worker_info["stub"].HealthCheck(inference_pb2.HealthRequest())
                worker_status.append({"worker": i, "status": response.status, "layers": f"{worker_info['assignment']['start_layer']}-{worker_info['assignment']['end_layer']-1}"})
            except:
                worker_status.append({"worker": i, "status": "offline", "layers": f"{worker_info['assignment']['start_layer']}-{worker_info['assignment']['end_layer']-1}"})
    
    return {
        "status": "healthy",
        "model": "mlx-community/Qwen3-1.7B-8bit",
        "workers": worker_status
    }

# Also run local worker gRPC service
class LocalWorkerService(inference_pb2_grpc.InferenceServiceServicer):
    """Local worker service for processing layers on coordinator."""
    
    def ProcessLayers(self, request, context):
        """Process transformer layers locally."""
        try:
            input_tensor = deserialize_tensor(request.input_tensor)
            
            with mx.stream(mx.gpu):
                hidden = input_tensor
                for i in range(request.start_layer, request.end_layer):
                    hidden = model.model.layers[i](hidden)
                mx.eval(hidden)
            
            response = inference_pb2.LayerResponseV2()
            response.output_tensor.CopyFrom(serialize_tensor(hidden))
            return response
            
        except Exception as e:
            logger.error(f"ProcessLayers error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    def HealthCheck(self, request, context):
        """Health check."""
        return inference_pb2.HealthResponse(status="healthy")
    
    def Forward(self, request, context):
        """Handle special forward passes (embeddings, final projection)."""
        try:
            if request.is_embedding:
                # Process embedding layer
                input_ids = mx.array(request.input_ids).reshape(1, -1)
                
                with mx.stream(mx.gpu):
                    embeddings = model.model.embed_tokens(input_ids)
                    mx.eval(embeddings)
                
                response = inference_pb2.ForwardResponse()
                response.output.CopyFrom(serialize_tensor(embeddings))
                return response
                
            elif request.is_final_projection:
                # Process final norm and projection
                input_tensor = deserialize_tensor(request.input_tensor)
                
                with mx.stream(mx.gpu):
                    # Apply final layer norm
                    normed = model.model.norm(input_tensor)
                    # Project to vocabulary
                    # Most MLX models use tied embeddings
                    if hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'as_linear'):
                        # Use tied embeddings (most common case)
                        logits = model.model.embed_tokens.as_linear(normed)
                    elif hasattr(model, 'lm_head') and model.lm_head is not None:
                        # Use separate lm_head if available
                        logits = model.lm_head(normed)
                    else:
                        logger.error(f"Cannot find output projection in LocalWorkerService")
                        raise AttributeError("Cannot find output projection layer")
                    mx.eval(logits)
                
                response = inference_pb2.ForwardResponse()
                response.output.CopyFrom(serialize_tensor(logits))
                return response
                
            else:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unknown forward request type")
                
        except Exception as e:
            logger.error(f"Forward error: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

async def serve_grpc():
    """Serve gRPC service for local worker."""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(LocalWorkerService(), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    logger.info("Local gRPC worker service started on port 50051")
    await server.wait_for_termination()

async def main():
    """Main entry point."""
    # Initialize coordinator
    initialize_coordinator()
    
    # Start both gRPC and FastAPI servers
    grpc_task = asyncio.create_task(serve_grpc())
    
    # Start FastAPI
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    logger.info("Starting FastAPI server on port 8100")
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())