#!/usr/bin/env python3
"""
Basic distributed inference - keeping it simple.
Run the full model through each device sequentially.
"""

import mlx.core as mx
from mlx_lm import load, generate
import grpc
from concurrent import futures
import time
import logging
import argparse
from typing import List, Dict, Any

# Import our proto files
from src.communication import inference_pb2, inference_pb2_grpc
from src.communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleWorker(inference_pb2_grpc.InferenceServiceServicer):
    """Simple worker that processes model layers."""
    
    def __init__(self, device_id: str, model_name: str, layer_start: int, layer_end: int):
        self.device_id = device_id
        self.model_name = model_name
        self.layer_start = layer_start
        self.layer_end = layer_end
        
        # Load the full model
        logger.info(f"Worker {device_id} loading model...")
        self.model, self.tokenizer = load(model_name)
        logger.info(f"Worker {device_id} ready to process layers {layer_start}-{layer_end}")
        
    def ProcessLayers(self, request, context):
        """Process assigned layers."""
        # Deserialize input
        metadata = {
            'shape': list(request.metadata.shape),
            'dtype': request.metadata.dtype,
            'compressed': request.metadata.compressed
        }
        hidden_states = deserialize_mlx_array(request.input_tensor, metadata)
        
        # Process through our assigned layers
        for i in range(self.layer_start, self.layer_end + 1):
            if i < len(self.model.model.layers):
                # Just call the layer directly
                layer_output = self.model.model.layers[i](hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
                mx.eval(hidden_states)  # Force computation
        
        # Serialize output
        output_data, output_metadata = serialize_mlx_array(hidden_states, compress=False)
        
        return inference_pb2.LayerResponse(
            request_id=request.request_id,
            output_tensor=output_data,
            metadata=inference_pb2.TensorMetadata(
                shape=output_metadata['shape'],
                dtype=output_metadata['dtype'],
                compressed=output_metadata.get('compressed', False)
            ),
            processing_time_ms=0  # Not tracking for now
        )


def run_worker(device_id: str, port: int, model_name: str, layer_start: int, layer_end: int):
    """Run a worker server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker = SimpleWorker(device_id, model_name, layer_start, layer_end)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(worker, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"Worker {device_id} listening on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


def run_coordinator(model_name: str, workers: List[Dict[str, Any]], prompt: str, max_tokens: int):
    """Run the coordinator that orchestrates inference."""
    # Load model on coordinator
    logger.info("Coordinator loading model...")
    model, tokenizer = load(model_name)
    
    # Connect to workers
    worker_stubs = {}
    for worker in workers:
        channel = grpc.insecure_channel(f"{worker['host']}:{worker['port']}")
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        worker_stubs[worker['id']] = {
            'stub': stub,
            'layer_start': worker['layer_start'],
            'layer_end': worker['layer_end']
        }
        logger.info(f"Connected to worker {worker['id']}")
    
    # First, let's test with single device inference
    logger.info(f"\nTesting single device inference with prompt: '{prompt}'")
    single_response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    logger.info(f"Single device response: {single_response}")
    
    # Now try distributed
    logger.info("\nNow testing distributed inference...")
    
    # Tokenize
    input_ids = mx.array(tokenizer.encode(prompt))
    if len(input_ids.shape) == 1:
        input_ids = input_ids[None, :]  # Add batch dimension
    
    # Get embeddings
    hidden_states = model.model.embed_tokens(input_ids)
    
    # Process through coordinator's layers (0-9)
    logger.info("Processing coordinator layers 0-9...")
    logger.info(f"Initial hidden states: shape={hidden_states.shape}, mean={mx.mean(hidden_states).item():.4f}")
    for i in range(10):
        layer_output = model.model.layers[i](hidden_states)
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
        else:
            hidden_states = layer_output
        mx.eval(hidden_states)
    logger.info(f"After coordinator layers: shape={hidden_states.shape}, mean={mx.mean(hidden_states).item():.4f}")
    
    # Send to workers
    for worker_id, worker_info in worker_stubs.items():
        logger.info(f"Sending to {worker_id} for layers {worker_info['layer_start']}-{worker_info['layer_end']}")
        
        # Serialize
        data, metadata = serialize_mlx_array(hidden_states, compress=False)
        request = inference_pb2.LayerRequest(
            request_id=f"test-{time.time()}",
            input_tensor=data,
            layer_indices=list(range(worker_info['layer_start'], worker_info['layer_end'] + 1)),
            metadata=inference_pb2.TensorMetadata(
                shape=metadata['shape'],
                dtype=metadata['dtype'],
                compressed=metadata.get('compressed', False)
            )
        )
        
        # Process
        response = worker_info['stub'].ProcessLayers(request, timeout=30)
        
        # Deserialize
        output_metadata = {
            'shape': list(response.metadata.shape),
            'dtype': response.metadata.dtype,
            'compressed': response.metadata.compressed
        }
        hidden_states = deserialize_mlx_array(response.output_tensor, output_metadata)
        logger.info(f"After {worker_id}: shape={hidden_states.shape}, mean={mx.mean(hidden_states).item():.4f}")
    
    # Apply final norm
    hidden_states = model.model.norm(hidden_states)
    
    # Get logits
    logits = model.model.embed_tokens.as_linear(hidden_states)
    logger.info(f"Logits shape: {logits.shape}, hidden_states shape: {hidden_states.shape}")
    
    # Generate tokens
    generated_ids = []
    for i in range(max_tokens):
        # Simple argmax sampling for testing
        next_token = mx.argmax(logits[0, -1, :], axis=-1)
        token_id = next_token.item()
        logger.info(f"Token {i}: {token_id} -> '{tokenizer.decode([token_id])}'")
        generated_ids.append(token_id)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode
    response = tokenizer.decode(generated_ids)
    logger.info(f"Distributed response: {response}")
    
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["coordinator", "worker"], required=True)
    parser.add_argument("--device-id", type=str, help="Device ID")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-8bit")
    parser.add_argument("--layer-start", type=int, help="Start layer for worker")
    parser.add_argument("--layer-end", type=int, help="End layer for worker")
    parser.add_argument("--prompt", default="Hello! How are you?")
    parser.add_argument("--max-tokens", type=int, default=30)
    args = parser.parse_args()
    
    if args.role == "worker":
        run_worker(args.device_id, args.port, args.model, args.layer_start, args.layer_end)
    else:
        # Define workers
        workers = [
            {"id": "mini2", "host": "169.254.149.32", "port": 50051, "layer_start": 10, "layer_end": 18},
            {"id": "m4", "host": "192.168.2.106", "port": 50051, "layer_start": 19, "layer_end": 27}
        ]
        run_coordinator(args.model, workers, args.prompt, args.max_tokens)


if __name__ == "__main__":
    main()