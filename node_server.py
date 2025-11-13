#!/usr/bin/env python3
"""
MLX Distributed Node Server - Exo Style
Each node runs independently and communicates via gRPC
"""
import argparse
import asyncio
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

import grpc
from concurrent import futures
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

import inference_service_pb2
import inference_service_pb2_grpc

# Add exo path for model imports
sys.path.insert(0, "/Users/mini1/Movies/exo")
from exo.inference.mlx.models.qwen2 import Model, ModelArgs
from exo.inference.shard import Shard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU as default
mx.set_default_device(mx.gpu)


class InferenceNode(inference_service_pb2_grpc.InferenceServiceServicer):
    def __init__(self, node_id: str, shard: Shard, next_node: Optional[str] = None):
        self.node_id = node_id
        self.shard = shard
        self.next_node = next_node
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.model_name = "mlx-community/Qwen3-1.7B-8bit"
        
    def load_model(self):
        """Load the sharded model"""
        logger.info(f"Node {self.node_id}: Loading layers {self.shard.start_layer}-{self.shard.end_layer}")
        
        # Get model path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_id = f"models--{self.model_name.replace('/', '--')}"
        model_path = cache_dir / model_id / "snapshots"
        model_path = sorted([d for d in model_path.iterdir() if d.is_dir()])[-1]
        
        # Load config
        with open(model_path / "config.json") as f:
            config = json.load(f)
        
        # Add shard info
        config["shard"] = {
            "model_id": self.model_name,
            "start_layer": self.shard.start_layer,
            "end_layer": self.shard.end_layer,
            "n_layers": self.shard.n_layers
        }
        
        # Create model
        args = ModelArgs.from_dict(config)
        self.model = Model(args)
        
        # Load weights
        import glob
        weight_files = sorted(glob.glob(str(model_path / "*.safetensors")))
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        
        # Filter weights for this shard
        filtered_weights = self.model.sanitize(weights)
        logger.info(f"Node {self.node_id}: Loading {len(filtered_weights)} weight tensors")
        
        # Load weights
        self.model.load_weights(list(filtered_weights.items()), strict=False)
        mx.eval(self.model.parameters())
        self.model.eval()
        
        # Load tokenizer if first or last shard
        if self.shard.is_first_layer() or self.shard.is_last_layer():
            self.tokenizer = load_tokenizer(model_path)
        
        # Create cache
        self.cache = make_prompt_cache(self.model)
        
        memory_gb = mx.get_active_memory() / 1e9
        logger.info(f"Node {self.node_id}: Model loaded, memory: {memory_gb:.2f} GB")
    
    async def ProcessTensor(self, request, context):
        """Process tensor through this shard"""
        # Deserialize input tensor
        tensor_np = np.frombuffer(request.tensor_data, dtype=request.dtype).reshape(request.shape)
        tensor = mx.array(tensor_np)
        
        # Process through model
        output = self.model(tensor, self.cache)
        mx.eval(output)
        
        # If this is the last shard, we're done
        is_final = self.shard.is_last_layer()
        
        # If not final and we have a next node, we'll forward it
        if not is_final and self.next_node:
            # The caller will handle forwarding to next node
            pass
        
        # Serialize output
        output_np = np.array(output)
        
        return inference_service_pb2.TensorResponse(
            tensor_data=output_np.tobytes(),
            shape=list(output_np.shape),
            dtype=str(output_np.dtype),
            is_final=is_final
        )
    
    async def HealthCheck(self, request, context):
        """Health check"""
        return inference_service_pb2.HealthStatus(
            healthy=True,
            shard=inference_service_pb2.Shard(
                model_id=self.model_name,
                start_layer=self.shard.start_layer,
                end_layer=self.shard.end_layer,
                n_layers=self.shard.n_layers
            ),
            memory_gb=mx.get_active_memory() / 1e9,
            node_id=self.node_id
        )


class NodeServer:
    def __init__(self, node_id: str, host: str, port: int, shard: Shard, next_node: Optional[str] = None):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.shard = shard
        self.next_node = next_node
        self.service = InferenceNode(node_id, shard, next_node)
        self.server = None
        self.next_node_stub = None
        
    async def start(self):
        """Start the gRPC server"""
        # Load model
        self.service.load_model()
        
        # Connect to next node if specified
        if self.next_node:
            channel = grpc.aio.insecure_channel(self.next_node)
            self.next_node_stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
            logger.info(f"Node {self.node_id}: Connected to next node at {self.next_node}")
        
        # Start gRPC server
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 256*1024*1024),
                ("grpc.max_receive_message_length", 256*1024*1024),
            ]
        )
        inference_service_pb2_grpc.add_InferenceServiceServicer_to_server(self.service, self.server)
        
        listen_addr = f"{self.host}:{self.port}"
        self.server.add_insecure_port(listen_addr)
        await self.server.start()
        
        logger.info(f"Node {self.node_id} started on {listen_addr}")
        logger.info(f"Shard: layers {self.shard.start_layer}-{self.shard.end_layer}")
        if self.next_node:
            logger.info(f"Next node: {self.next_node}")
        
        # Keep server running
        await self.server.wait_for_termination()
    
    async def process_prompt(self, prompt: str) -> str:
        """Process a prompt through the pipeline (only for first node)"""
        if not self.shard.is_first_layer():
            raise ValueError("Only the first node can process prompts")
        
        # Tokenize
        tokens = self.service.tokenizer.encode(prompt)
        input_tensor = mx.array(tokens).reshape(1, -1)
        
        # Process through our shard
        output = self.service.model(input_tensor, self.service.cache)
        mx.eval(output)
        
        # If we're also the last shard, we're done
        if self.shard.is_last_layer():
            return self._sample_and_decode(output)
        
        # Otherwise, forward to next node
        if self.next_node_stub:
            output_np = np.array(output)
            request = inference_service_pb2.TensorRequest(
                tensor_data=output_np.tobytes(),
                shape=list(output_np.shape),
                dtype=str(output_np.dtype),
                request_id="test",
                is_prompt=True
            )
            
            response = await self.next_node_stub.ProcessTensor(request)
            
            # If this is the final output, decode it
            if response.is_final:
                result_np = np.frombuffer(response.tensor_data, dtype=response.dtype).reshape(response.shape)
                result = mx.array(result_np)
                return self._sample_and_decode(result)
        
        return "Error: Pipeline incomplete"
    
    def _sample_and_decode(self, logits: mx.array) -> str:
        """Sample token and decode (for testing)"""
        sampler = make_sampler(0.7)
        logits = logits[:, -1, :]
        token = sampler(logits)
        mx.eval(token)
        token_id = int(token.item())
        return self.service.tokenizer.decode([token_id])


async def main():
    parser = argparse.ArgumentParser(description="MLX Distributed Node Server")
    parser.add_argument("--node-id", required=True, help="Node ID (e.g., mini1, mini2)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--start-layer", type=int, required=True, help="Start layer")
    parser.add_argument("--end-layer", type=int, required=True, help="End layer")
    parser.add_argument("--next-node", help="Address of next node (e.g., 192.168.5.2:50051)")
    args = parser.parse_args()
    
    # Create shard
    shard = Shard(
        model_id="mlx-community/Qwen3-1.7B-8bit",
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        n_layers=28
    )
    
    # Create and start server
    server = NodeServer(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        shard=shard,
        next_node=args.next_node
    )
    
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())