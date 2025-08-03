#!/usr/bin/env python3
"""
Debug distributed inference gibberish output.
Following MLX/gRPC best practices for debugging.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate
import asyncio
import logging
import time
from typing import List, Tuple

import sys
sys.path.insert(0, 'src')

from communication import inference_pb2_grpc, inference_pb2
from communication.dns_resolver import resolve_grpc_target
from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
from core.config import ClusterConfig
import grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedDebugger:
    def __init__(self, config_path: str = 'config/cluster_config.yaml'):
        self.config = ClusterConfig.from_yaml(config_path)
        self.model = None
        self.tokenizer = None
        self.worker_connections = {}
        
    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model.name}")
        self.model, self.tokenizer = load(self.config.model.name)
        logger.info(f"Model loaded with {len(self.model.model.layers)} layers")
        
    def connect_to_workers(self):
        """Connect to worker devices."""
        for device in self.config.devices:
            if device.device_id == self.config.coordinator_device_id:
                continue
                
            try:
                target = resolve_grpc_target(device.hostname, device.grpc_port)
                channel = grpc.insecure_channel(
                    target,
                    options=[
                        ('grpc.max_send_message_length', 500 * 1024 * 1024),
                        ('grpc.max_receive_message_length', 500 * 1024 * 1024),
                    ]
                )
                stub = inference_pb2_grpc.InferenceServiceStub(channel)
                
                # Test connection
                health_response = stub.HealthCheck(inference_pb2.Empty(), timeout=5.0)
                if health_response.healthy:
                    self.worker_connections[device.device_id] = {
                        'stub': stub,
                        'channel': channel,
                        'layers': self.config.model.get_device_layers(device.device_id)
                    }
                    logger.info(f"✅ Connected to {device.device_id}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to {device.device_id}: {e}")
    
    def log_tensor_stats(self, tensor: mx.array, name: str):
        """Log statistics about a tensor."""
        stats = {
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'mean': float(mx.mean(tensor).item()),
            'std': float(mx.std(tensor).item()),
            'min': float(mx.min(tensor).item()),
            'max': float(mx.max(tensor).item()),
            'has_nan': bool(mx.any(mx.isnan(tensor)).item()),
            'has_inf': bool(mx.any(mx.isinf(tensor)).item()),
        }
        logger.info(f"📊 {name}: {stats}")
        return stats
    
    def test_single_node_inference(self, prompt: str) -> Tuple[mx.array, str]:
        """Run inference on single node (no distribution)."""
        logger.info("🔍 Testing single-node inference...")
        
        # Tokenize
        input_ids = mx.array(self.tokenizer.encode(prompt))
        logger.info(f"Input shape: {input_ids.shape}")
        
        # Get embeddings
        embeddings = self.model.model.embed_tokens(input_ids[None, :])
        self.log_tensor_stats(embeddings, "Embeddings")
        
        # Process through all layers locally
        hidden_states = embeddings
        for i, layer in enumerate(self.model.model.layers):
            hidden_states = layer(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            
            if i % 5 == 0:  # Log every 5th layer
                self.log_tensor_stats(hidden_states, f"Layer {i}")
        
        # Final norm and output
        hidden_states = self.model.model.norm(hidden_states)
        self.log_tensor_stats(hidden_states, "After norm")
        
        # Get logits
        logits = self.model.model.embed_tokens.as_linear(hidden_states)
        self.log_tensor_stats(logits, "Logits")
        
        # Generate using simple approach
        response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=20)
        if response.startswith(prompt):
            response = response[len(prompt):]
        
        logger.info(f"Single-node response: {response}")
        return logits, response
    
    def test_distributed_inference(self, prompt: str) -> Tuple[mx.array, str]:
        """Run distributed inference."""
        logger.info("🌐 Testing distributed inference...")
        
        # Tokenize
        input_ids = mx.array(self.tokenizer.encode(prompt))
        
        # Get embeddings
        embeddings = self.model.model.embed_tokens(input_ids[None, :])
        self.log_tensor_stats(embeddings, "Embeddings (distributed)")
        
        # Process coordinator layers
        hidden_states = embeddings
        coordinator_layers = self.config.model.get_device_layers(self.config.coordinator_device_id)
        
        for layer_idx in coordinator_layers:
            hidden_states = self.model.model.layers[layer_idx](hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        
        self.log_tensor_stats(hidden_states, f"After coordinator layers {coordinator_layers}")
        
        # Process through workers
        for worker_id, worker_info in sorted(self.worker_connections.items(), 
                                            key=lambda x: min(x[1]['layers'])):
            stub = worker_info['stub']
            layers = worker_info['layers']
            
            logger.info(f"Sending to {worker_id} for layers {layers}")
            
            # Serialize
            data, metadata = serialize_mlx_array(hidden_states, compress=False)
            
            # Create request
            request = inference_pb2.LayerRequest(
                request_id=f"debug-{time.time()}",
                input_tensor=data,
                layer_indices=layers,
                metadata=inference_pb2.TensorMetadata(
                    shape=metadata['shape'],
                    dtype=metadata['dtype'],
                    compressed=metadata.get('compressed', False)
                )
            )
            
            # Process
            response = stub.ProcessLayers(request, timeout=30.0)
            
            # Deserialize
            output_metadata = {
                'shape': list(response.metadata.shape),
                'dtype': response.metadata.dtype,
                'compressed': response.metadata.compressed
            }
            hidden_states = deserialize_mlx_array(response.output_tensor, output_metadata)
            
            self.log_tensor_stats(hidden_states, f"After {worker_id} layers {layers}")
        
        # Final norm and output
        hidden_states = self.model.model.norm(hidden_states)
        self.log_tensor_stats(hidden_states, "After norm (distributed)")
        
        # Get logits
        logits = self.model.model.embed_tokens.as_linear(hidden_states)
        self.log_tensor_stats(logits, "Logits (distributed)")
        
        # Simple generation
        next_token = mx.argmax(logits[0, -1, :])
        response_ids = [next_token.item()]
        for _ in range(19):  # Generate 19 more tokens
            next_token = mx.argmax(logits[0, -1, :])
            response_ids.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        response = self.tokenizer.decode(response_ids)
        logger.info(f"Distributed response: {response}")
        return logits, response
    
    def compare_outputs(self, single_logits: mx.array, dist_logits: mx.array):
        """Compare single-node vs distributed outputs."""
        logger.info("🔍 Comparing outputs...")
        
        # Calculate differences
        diff = mx.abs(single_logits - dist_logits)
        max_diff = mx.max(diff).item()
        mean_diff = mx.mean(diff).item()
        
        logger.info(f"Max difference: {max_diff}")
        logger.info(f"Mean difference: {mean_diff}")
        
        # Check if outputs match
        threshold = 1e-3
        if max_diff < threshold:
            logger.info("✅ Outputs match within threshold!")
        else:
            logger.error(f"❌ Outputs differ significantly! Max diff: {max_diff}")
            
            # Find where they differ most
            flat_diff = diff.reshape(-1)
            top_diffs = mx.argsort(flat_diff)[-10:]
            logger.info("Top 10 differences:")
            for idx in top_diffs:
                logger.info(f"  Index {idx}: diff = {flat_diff[idx].item()}")
    
    def run_debug_session(self):
        """Run full debug session."""
        self.load_model()
        self.connect_to_workers()
        
        if not self.worker_connections:
            logger.error("No workers connected! Cannot run distributed test.")
            return
        
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Adaptive Multi-Teacher Multi-level Knowledge Distillation",
            "The capital of France is",
        ]
        
        for prompt in test_prompts:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing prompt: '{prompt}'")
            logger.info(f"{'='*60}")
            
            # Single node
            single_logits, single_response = self.test_single_node_inference(prompt)
            
            # Distributed
            dist_logits, dist_response = self.test_distributed_inference(prompt)
            
            # Compare
            self.compare_outputs(single_logits, dist_logits)
            
            logger.info(f"\n📝 Results:")
            logger.info(f"Single-node: {single_response}")
            logger.info(f"Distributed: {dist_response}")
            logger.info(f"Match: {single_response == dist_response}")


if __name__ == "__main__":
    import os
    
    # Enable gRPC debugging if requested
    if os.getenv("GRPC_DEBUG"):
        os.environ["GRPC_TRACE"] = "all"
        os.environ["GRPC_VERBOSITY"] = "DEBUG"
    
    debugger = DistributedDebugger()
    debugger.run_debug_session()