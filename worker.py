#!/usr/bin/env python3
"""
Worker node for distributed MLX inference.
Loads partial model layers and processes them via gRPC.
"""
import os
import sys
import grpc
import logging
import asyncio
from concurrent import futures
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_causal_mask

# Force reload proto modules to avoid stale imports
import importlib
proto_modules = ['src.communication.inference_pb2', 'src.communication.inference_pb2_grpc', 
                 'src.communication.tensor_utils', 'src.communication']
for module_name in proto_modules:
    if module_name in sys.modules:
        del sys.modules[module_name]

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

print("WORKER VERSION: 2.0 - Debug ProcessLayers", flush=True)

# Global model state
model = None
tokenizer = None
device_id = None
assigned_layers = None

class WorkerService(inference_pb2_grpc.InferenceServiceServicer):
    """Worker service for processing model layers."""
    
    def __init__(self, worker_id: int, start_layer: int, end_layer: int, is_first: bool, is_last: bool):
        self.worker_id = worker_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.is_first = is_first
        self.is_last = is_last
        logger.info(f"Worker {worker_id} initialized for layers {start_layer}-{end_layer-1}")
    
    def ProcessLayers(self, request, context):
        """Process transformer layers."""
        try:
            logger.info(f"ProcessLayers called - start_layer={request.start_layer}, end_layer={request.end_layer}")
            logger.info(f"Input tensor shape: {list(request.input_tensor.shape)}, dtype={request.input_tensor.dtype}")
            
            # Deserialize input tensor
            input_tensor = deserialize_tensor(request.input_tensor)
            logger.info(f"Deserialized tensor shape: {input_tensor.shape}, mean: {input_tensor.mean():.6f}, std: {input_tensor.std():.6f}")
            
            # Create attention mask locally based on sequence length
            T = input_tensor.shape[1]
            if T > 1:
                attention_mask = create_causal_mask(T, offset=0)
                logger.info(f"Created local causal mask with shape: {attention_mask.shape}")
            else:
                attention_mask = None
                logger.info("Single token - no mask needed")
            
            # Check input validity - allow higher std for this quantized model
            input_std = float(input_tensor.std())
            if input_std > 200.0:  # Increased threshold for Qwen3-1.7B-8bit
                logger.error(f"Input tensor std deviation {input_std:.2f} is too high!")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Input numerical instability: std={input_std:.2f}")
            
            # Process layers with MLX GPU stream
            with mx.stream(mx.gpu):
                hidden = input_tensor
                
                # CRITICAL: Ensure input is fully loaded before processing
                mx.eval(hidden)
                mx.synchronize()
                
                # Create cache for layers
                cache = [None] * (request.end_layer - request.start_layer)
                
                for idx, i in enumerate(range(request.start_layer, request.end_layer)):
                    if i < len(model.model.layers):
                        # Pass mask and cache
                        hidden = model.model.layers[i](hidden, attention_mask, cache[idx])
                        # Force evaluation after each layer
                        mx.eval(hidden)
                        
                        # CRITICAL: Synchronize to prevent race conditions
                        mx.synchronize()
                        
                        # Check for numerical instability after each layer
                        layer_std = float(hidden.std())
                        if layer_std > 200.0:  # Increased threshold for Qwen3-1.7B-8bit
                            logger.error(f"Layer {i} produced unstable output: std={layer_std:.2f}")
                            context.abort(grpc.StatusCode.INTERNAL, f"Layer {i} numerical instability: std={layer_std:.2f}")
                    else:
                        logger.warning(f"Layer {i} out of bounds (model has {len(model.model.layers)} layers)")
                
                # Ensure final result is evaluated and synchronized
                mx.eval(hidden)
                mx.synchronize()
            
            # Final validation
            output_std = float(hidden.std())
            logger.info(f"Output tensor stats - shape: {hidden.shape}, mean: {hidden.mean():.6f}, std: {output_std:.6f}")
            
            if output_std > 200.0:  # Increased threshold for Qwen3-1.7B-8bit
                logger.error(f"Output tensor std deviation {output_std:.2f} is too high!")
                context.abort(grpc.StatusCode.INTERNAL, f"Output numerical instability: std={output_std:.2f}")
            
            # Serialize output
            response = inference_pb2.LayerResponseV2()
            response.output_tensor.CopyFrom(serialize_tensor(hidden))
            
            logger.info(f"Processed layers successfully")
            return response
            
        except Exception as e:
            logger.error(f"ProcessLayers error: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
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
                        logger.error(f"Cannot find output projection. Model type: {type(model)}")
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
    
    def HealthCheck(self, request, context):
        """Health check endpoint."""
        mem_info = mx.metal.device_info() or {}
        used = mem_info.get('used_memory', 0)
        total = mem_info.get('total_memory', 0)
        
        logger.info(f"Health check - Memory: {used/1e9:.2f}GB/{total/1e9:.2f}GB")
        return inference_pb2.HealthResponse(
            status="healthy",
            message=f"Worker {self.worker_id} ready, layers {self.start_layer}-{self.end_layer-1}"
        )

def initialize_worker(worker_id: int, total_workers: int):
    """Initialize worker with model layers."""
    global model, tokenizer, device_id, assigned_layers
    
    device_id = worker_id
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    
    logger.info(f"Loading model {model_name} on worker {worker_id} (total_workers={total_workers})")
    # CRITICAL: Don't use lazy loading - load the full model to ensure weights are correct
    model, tokenizer = load(model_name, lazy=False)
    
    # Calculate layer assignment FIRST before using the variables
    total_layers = len(model.model.layers)
    layers_per_worker = total_layers // total_workers
    remainder = total_layers % total_workers
    
    start_layer = 0
    for i in range(worker_id):
        start_layer += layers_per_worker + (1 if i < remainder else 0)
    
    num_layers = layers_per_worker + (1 if worker_id < remainder else 0)
    end_layer = start_layer + num_layers
    
    assigned_layers = (start_layer, end_layer)
    is_first = worker_id == 0
    is_last = worker_id == total_workers - 1
    
    # CRITICAL: Force model evaluation to ensure weights are loaded properly
    logger.info(f"Testing model layers {start_layer} to {end_layer-1}...")
    with mx.stream(mx.gpu):
        # Test with a realistic input
        test_tokens = [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
        test_input = mx.array([test_tokens[:5]])  # Use first 5 tokens
        
        # Get embeddings (we need this even if we're not the first worker)
        test_hidden = model.model.embed_tokens(test_input)
        mx.eval(test_hidden)
        logger.info(f"Test embeddings shape: {test_hidden.shape}, std: {test_hidden.std():.6f}")
        
        # Create attention mask for testing
        test_mask = create_attention_mask(test_hidden, cache=None)
        test_cache = [None] * len(model.model.layers)
        
        # If we're not the first worker, we need to process through earlier layers
        # to get a proper input state
        if start_layer > 0:
            logger.info(f"Processing through layers 0-{start_layer-1} to prepare input state...")
            for i in range(start_layer):
                test_hidden = model.model.layers[i](test_hidden, test_mask, test_cache[i])
                mx.eval(test_hidden)
            logger.info(f"Input preparation complete, std: {test_hidden.std():.6f}")
        
        # Now test our assigned layers
        logger.info(f"Testing assigned layers {start_layer}-{end_layer-1}...")
        for i in range(start_layer, end_layer):
            test_hidden = model.model.layers[i](test_hidden, test_mask, test_cache[i])
            mx.eval(test_hidden)
            if test_hidden.std() > 200:  # Increased threshold - high std is normal with mask
                logger.warning(f"Layer {i} produces high std: {test_hidden.std():.6f} (normal with attention mask)")
        
        logger.info(f"Model test complete, final std: {test_hidden.std():.6f}")
    
    # Log memory usage
    mem_info = mx.metal.device_info() or {}
    used = mem_info.get('used_memory', 0)
    total = mem_info.get('total_memory', 0)
    logger.info(f"Model loaded. Memory: {used/1e9:.2f}GB/{total/1e9:.2f}GB")
    logger.info(f"Assigned layers: {start_layer}-{end_layer-1} (is_first={is_first}, is_last={is_last})")
    logger.info(f"Worker {worker_id} ready with {num_layers} layers out of {total_layers} total")
    
    return WorkerService(worker_id, start_layer, end_layer, is_first, is_last)

async def serve():
    """Start gRPC server."""
    # Get worker configuration from command line or environment
    worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv('WORKER_ID', '1'))
    total_workers = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.getenv('TOTAL_WORKERS', '3'))
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 50051
    
    # Initialize worker
    service = initialize_worker(worker_id, total_workers)
    
    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.max_receive_message_length', 500 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
        ]
    )
    
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(service, server)
    
    # Bind to all interfaces on specified port
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    await server.start()
    logger.info(f"Worker {worker_id} gRPC server started on port {port}")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
        await server.stop(5)

if __name__ == "__main__":
    asyncio.run(serve())