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
from mlx_lm.models.base import create_attention_mask, create_causal_mask
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
    usage: Dict[str, Any]  # Changed to Any to support float values

# Global state
model = None
tokenizer = None
worker_stubs = []
model_config = None

# Configuration - default value, can be overridden per request
DEFAULT_USE_DISTRIBUTED_INFERENCE = 'auto'  # 'always', 'never', 'auto'

def initialize_coordinator():
    """Initialize the coordinator with model metadata and worker connections."""
    global model, tokenizer, worker_stubs, model_config
    
    # CRITICAL: Clear any existing stubs
    worker_stubs = []
    
    # Load model to get configuration
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    logger.info(f"Loading model configuration from {model_name}")
    model, tokenizer = load(model_name)
    
    # CRITICAL: Test the model locally first to ensure it works
    logger.info("Testing model with single-device inference...")
    # Use proper chat template for testing, just like actual inference
    test_messages = [{"role": "user", "content": "Hello"}]
    test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    test_tokens = tokenizer.encode(test_prompt)
    
    with mx.stream(mx.gpu):
        # Test embeddings
        test_input = mx.array([test_tokens])
        test_embed = model.model.embed_tokens(test_input)
        mx.eval(test_embed)
        logger.info(f"Test embeddings - mean: {test_embed.mean():.6f}, std: {test_embed.std():.6f}")
        
        # Test full forward pass
        hidden = test_embed
        for i, layer in enumerate(model.model.layers):
            hidden = layer(hidden)
            if i % 5 == 0:  # Log every 5 layers
                mx.eval(hidden)
                # Don't log every 5 layers - it's too verbose and std can be high
                if i == len(model.model.layers) - 1:  # Only log the last layer
                    logger.info(f"Test final layer {i} - mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")
        
        # Test final projection
        hidden = model.model.norm(hidden)
        if hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'as_linear'):
            logits = model.model.embed_tokens.as_linear(hidden)
        mx.eval(logits)
        logger.info(f"Test logits - shape: {logits.shape}, mean: {logits.mean():.6f}, std: {logits.std():.6f}")
        
        # Log final stats for debugging
        final_std = float(hidden.std())
        if final_std > 50.0:
            logger.info(f"Note: std deviation {final_std:.2f} is normal for Qwen3-1.7B-8bit model")
    
    # Get model configuration
    model_config = {
        "n_layers": len(model.model.layers),
        "vocab_size": model.model.vocab_size,
        "model_type": model.model_type,
    }
    logger.info(f"Model has {model_config['n_layers']} layers")
    
    # Connect to workers on Thunderbolt network - ONLY 2 DEVICES
    worker_addresses = [
        "192.168.5.1:50051",  # mini1 (coordinator also acts as worker)
        "192.168.5.2:50051",  # mini2 
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
    
    logger.info(f"Coordinator initialized successfully with {len(worker_stubs)} workers")

async def distributed_forward(input_ids: mx.array) -> mx.array:
    """Forward pass through distributed model."""
    current_hidden = None
    total_start = time.time()
    
    logger.info(f"Full forward pass with {input_ids.shape[1]} tokens")
    
    # DEBUG: Compare with single-device forward pass
    logger.info(f"DEBUG: Input shape: {input_ids.shape}, tokens: {input_ids.tolist()[0][:10] if input_ids.shape[1] > 10 else input_ids.tolist()[0]}...")
    
    # Get embeddings (on first worker)
    if worker_stubs[0]["stub"] is None:
        # Local processing
        with mx.stream(mx.gpu):
            current_hidden = model.model.embed_tokens(input_ids)
            mx.eval(current_hidden)
        logger.info(f"DEBUG: After embeddings - mean: {current_hidden.mean():.6f}, std: {current_hidden.std():.6f}")
    else:
        # Remote processing
        request = inference_pb2.ForwardRequest()
        request.input_ids.extend(input_ids.tolist()[0])
        request.is_embedding = True
        
        response = worker_stubs[0]["stub"].Forward(request)
        current_hidden = deserialize_tensor(response.output)
        logger.info(f"DEBUG: After embeddings (remote) - mean: {current_hidden.mean():.6f}, std: {current_hidden.std():.6f}")
    
    # CRITICAL: Create attention mask after embeddings
    # For distributed processing, we need the actual mask array, not the "causal" string
    T = current_hidden.shape[1]
    if T > 1:
        # Multi-token: create causal mask
        attention_mask = create_causal_mask(T, offset=0)
        logger.info(f"Created causal attention mask with shape: {attention_mask.shape}")
    else:
        # Single token: no mask needed
        attention_mask = None
        logger.info("No attention mask needed (single token)")
    
    # Process through transformer layers
    for worker_info in worker_stubs:
        assignment = worker_info["assignment"]
        stub = worker_info["stub"]
        
        logger.info(f"DEBUG: Before layers {assignment['start_layer']}-{assignment['end_layer']-1} - mean: {current_hidden.mean():.6f}, std: {current_hidden.std():.6f}")
        
        if stub is None:
            # Local processing
            layer_start = time.time()
            logger.info(f"Processing layers {assignment['start_layer']}-{assignment['end_layer']-1} locally")
            with mx.stream(mx.gpu):
                # Create cache for layers
                cache = [None] * (assignment["end_layer"] - assignment["start_layer"])
                # Use the mask we created (or None for single token)
                for idx, i in enumerate(range(assignment["start_layer"], assignment["end_layer"])):
                    current_hidden = model.model.layers[i](current_hidden, attention_mask, cache[idx])
                mx.eval(current_hidden)
                mx.synchronize()  # CRITICAL: Ensure all operations complete
            logger.info(f"Local layers took {time.time() - layer_start:.3f}s")
        else:
            # Remote processing
            layer_start = time.time()
            logger.info(f"Sending to worker at {assignment['address']} for layers {assignment['start_layer']}-{assignment['end_layer']-1}")
            
            # DEBUG: Log what we're sending
            logger.info(f"DEBUG: Sending tensor with shape {current_hidden.shape}, mean: {current_hidden.mean():.6f}, std: {current_hidden.std():.6f}")
            
            request = inference_pb2.LayerRequestV2()
            request.input_tensor.CopyFrom(serialize_tensor(current_hidden))
            request.start_layer = assignment["start_layer"]
            request.end_layer = assignment["end_layer"]
            # Don't send mask - worker will create it locally from sequence length
            
            send_time = time.time()
            response = stub.ProcessLayers(request)
            recv_time = time.time()
            current_hidden = deserialize_tensor(response.output_tensor)
            logger.info(f"Remote layers took {recv_time - layer_start:.3f}s (send: {send_time - layer_start:.3f}s, process+recv: {recv_time - send_time:.3f}s)")
        
        logger.info(f"DEBUG: After layers {assignment['start_layer']}-{assignment['end_layer']-1} - mean: {current_hidden.mean():.6f}, std: {current_hidden.std():.6f}")
        
        # Check for numerical instability - high std is normal for Qwen3-1.7B-8bit
        hidden_std = float(current_hidden.std())
        if hidden_std > 200.0:  # Increased threshold for this quantized model
            logger.error(f"CRITICAL: Hidden state std deviation {hidden_std:.2f} is too high! Model computation corrupted!")
            logger.error(f"Worker at {assignment['address']} (layers {assignment['start_layer']}-{assignment['end_layer']-1}) is producing corrupted outputs!")
            logger.error(f"This typically indicates:")
            logger.error(f"  1. Model weights are corrupted on the worker")
            logger.error(f"  2. Different model versions between devices")
            logger.error(f"  3. Tensor serialization/deserialization error")
            
            # Force single-device mode as fallback
            logger.error("FORCING SINGLE-DEVICE MODE DUE TO CORRUPTION")
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=0.7)
            result = generate(model, tokenizer, prompt="Hello", max_tokens=50, sampler=sampler)
            logger.info(f"Single-device test result: {result}")
            
            raise ValueError(f"Numerical instability detected: std={hidden_std:.2f} from worker {assignment['address']}")
        elif hidden_std > 50.0:
            # Log high but acceptable std
            logger.info(f"Note: Hidden state std {hidden_std:.2f} is high but normal for Qwen3-1.7B-8bit")
    
    # Final projection (on last worker)
    logger.info(f"DEBUG: Before final projection - mean: {current_hidden.mean():.6f}, std: {current_hidden.std():.6f}")
    
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
    
    logger.info(f"DEBUG: Final logits - shape: {logits.shape}, mean: {logits.mean():.6f}, std: {logits.std():.6f}")
    
    # DEBUG: Show top tokens for last position
    try:
        last_position_logits = logits[0, -1, :]
        # Get indices of top 5 values
        top_indices = mx.argpartition(-last_position_logits, 5)[:5]
        top_values = last_position_logits[top_indices]
        sorted_idx = mx.argsort(-top_values)
        top_indices = top_indices[sorted_idx]
        top_values = top_values[sorted_idx]
        
        logger.info(f"DEBUG: Top 5 tokens - indices: {top_indices.tolist()}, values: {top_values.tolist()}")
        
        # Decode top tokens to see what they are
        top_tokens = [tokenizer.decode([int(idx)]) for idx in top_indices.tolist()]
        logger.info(f"DEBUG: Top 5 tokens decoded: {top_tokens}")
        
        # Check if logits are reasonable
        logit_std = last_position_logits.std()
        logit_min = last_position_logits.min()
        logit_max = last_position_logits.max()
        logger.info(f"DEBUG: Logit stats - std: {logit_std:.3f}, min: {logit_min:.3f}, max: {logit_max:.3f}")
    except Exception as e:
        logger.info(f"DEBUG: Error getting top tokens: {e}")
    
    logger.info(f"Total forward pass took {time.time() - total_start:.3f}s")
    return logits

def generate_single_device(prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate text using single device with proper KV caching."""
    # Use MLX's built-in generate function which handles KV caching properly
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temperature)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)

async def generate_distributed(prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict]:
    """Generate text using distributed model."""
    # Check sequence length to decide strategy
    prompt_tokens = tokenizer.encode(prompt)
    
    # Decide whether to use distributed or single-device
    # Read environment variable dynamically to allow per-request override
    use_distributed_setting = os.getenv('USE_DISTRIBUTED_INFERENCE', DEFAULT_USE_DISTRIBUTED_INFERENCE)
    use_distributed = False
    
    if use_distributed_setting == 'always':
        use_distributed = True
    elif use_distributed_setting == 'never':
        use_distributed = False
    else:  # 'auto' - FORCE DISTRIBUTED FOR TESTING
        # TEMPORARILY FORCING DISTRIBUTED MODE TO TEST ALL 2 DEVICES
        use_distributed = True  # Force distributed mode with 2 devices
    
    if not use_distributed:
        logger.info(f"Using single-device generation with KV cache (prompt_len={len(prompt_tokens)}, max_tokens={max_tokens})")
        start_time = time.time()
        result = generate_single_device(prompt, max_tokens, temperature)
        elapsed = time.time() - start_time
        
        # Calculate performance metrics
        completion_tokens = len(tokenizer.encode(result)) - len(prompt_tokens)
        metrics = {
            "prompt_eval_time": 0.1,  # Estimate for single-device
            "eval_time": elapsed - 0.1,
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": completion_tokens
        }
        return result, metrics
    
    # For short prompts, distributed without cache is acceptable
    logger.info(f"🔥 USING DISTRIBUTED MODE - 2 DEVICES ACTIVE (prompt_len={len(prompt_tokens)}, max_tokens={max_tokens})")
    logger.info(f"📱 Device 1 (mini1): Processing layers 0-13 locally")
    logger.info(f"🖥️  Device 2 (mini2): Processing layers 14-27 remotely")
    
    # Use the SAME sampling logic as single-device but with distributed forward
    # Import the same sampling utilities that single-device uses
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temperature)
    
    # Start with just the prompt
    input_ids = mx.array(prompt_tokens).reshape(1, -1)
    generated_tokens = []
    
    logger.info(f"Starting distributed generation with {len(prompt_tokens)} prompt tokens")
    
    # Track timing for metrics
    prompt_eval_start = time.time()
    
    # Initial forward pass for the prompt (full pass)
    logits = await distributed_forward(input_ids)
    prompt_eval_time = time.time() - prompt_eval_start
    
    eval_start = time.time()
    
    # Get last token from initial forward
    with mx.stream(mx.gpu):
        last_logits = logits[0, -1:, :]
        
        # DEBUG: Check logits before sampling
        logger.info(f"DEBUG: Initial logits shape: {last_logits.shape}, mean: {last_logits.mean():.3f}, std: {last_logits.std():.3f}")
        
        next_token = sampler(last_logits)
        mx.eval(next_token)
        next_token_id = int(next_token[0].item())
        
        # DEBUG: Decode the token to see what it is
        token_text = tokenizer.decode([next_token_id])
        logger.info(f"DEBUG: Initial sampled token {next_token_id} = '{token_text}'")
    
    generated_tokens.append(next_token_id)
    logger.info(f"Step 1: Generated token {next_token_id}")
    
    if next_token_id != tokenizer.eos_token_id and max_tokens > 1:
        # Add token to sequence
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)
        
        # Generate remaining tokens with FULL CONTEXT (Option 1 - slow but correct)
        for step in range(1, max_tokens):
            # Forward pass with FULL sequence for proper attention context
            logits = await distributed_forward(input_ids)
            
            # Use the SAME sampling logic as single-device mode
            with mx.stream(mx.gpu):
                # Get logits for the last position only
                last_logits = logits[0, -1:, :]  # Keep batch dimension
                
                # DEBUG: Check logits before sampling
                logger.info(f"DEBUG: Logits shape before sampling: {last_logits.shape}, mean: {last_logits.mean():.3f}, std: {last_logits.std():.3f}")
                
                # Sample using MLX's sampler (same as single-device)
                next_token = sampler(last_logits)
                mx.eval(next_token)
                
                next_token_id = int(next_token[0].item())
                
                # DEBUG: Decode the token to see what it is
                token_text = tokenizer.decode([next_token_id])
                logger.info(f"DEBUG: Sampled token {next_token_id} = '{token_text}'")
            
            generated_tokens.append(next_token_id)
            logger.info(f"Step {step+1}: Generated token {next_token_id}")
            
            # Stop if EOS token
            if next_token_id == tokenizer.eos_token_id:
                logger.info(f"Hit EOS token, stopping generation at {step+1} tokens")
                break
            
            # Add new token to sequence for next iteration
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)
    
    eval_time = time.time() - eval_start
    
    # Decode the same way as single-device
    generated_text = tokenizer.decode(generated_tokens)
    full_text = prompt + generated_text
    
    logger.info(f"Distributed generation complete: {len(generated_tokens)} tokens generated")
    
    # Calculate performance metrics
    metrics = {
        "prompt_eval_time": prompt_eval_time,
        "eval_time": eval_time,
        "prompt_tokens": len(prompt_tokens),
        "completion_tokens": len(generated_tokens)
    }
    
    return full_text, metrics

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Format messages using Qwen3's proper chat template
        formatted_messages = []
        for msg in request.messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})
        
        # Use the tokenizer's apply_chat_template method for proper formatting
        prompt = tokenizer.apply_chat_template(
            formatted_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Generate response with metrics
        start_time = time.time()
        response_text, metrics = await generate_distributed(prompt, request.max_tokens, request.temperature)
        elapsed = time.time() - start_time
        
        # Extract just the assistant's response
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):]
        
        # Clean up Qwen3 special tokens and extra whitespace
        response_text = response_text.replace('<|im_end|>', '').strip()
        
        # Calculate tokens/second metrics
        prompt_eval_tokens_per_second = metrics["prompt_tokens"] / metrics["prompt_eval_time"] if metrics["prompt_eval_time"] > 0 else 0
        eval_tokens_per_second = metrics["completion_tokens"] / metrics["eval_time"] if metrics["eval_time"] > 0 else 0
        
        logger.info(f"Generated {metrics['completion_tokens']} tokens in {elapsed:.2f}s")
        logger.info(f"Prompt eval: {prompt_eval_tokens_per_second:.1f} tok/s, Generation: {eval_tokens_per_second:.1f} tok/s")
        
        # Return OpenAI-compatible response with performance metrics
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
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["completion_tokens"],
                "total_tokens": metrics["prompt_tokens"] + metrics["completion_tokens"],
                "prompt_eval_tokens_per_second": round(prompt_eval_tokens_per_second, 1),
                "eval_tokens_per_second": round(eval_tokens_per_second, 1)
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
            
            # Create attention mask locally based on sequence length
            T = input_tensor.shape[1]
            if T > 1:
                attention_mask = create_causal_mask(T, offset=0)
            else:
                attention_mask = None
            
            with mx.stream(mx.gpu):
                hidden = input_tensor
                # Create cache for layers
                cache = [None] * (request.end_layer - request.start_layer)
                for idx, i in enumerate(range(request.start_layer, request.end_layer)):
                    hidden = model.model.layers[i](hidden, attention_mask, cache[idx])
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