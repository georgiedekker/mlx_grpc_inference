#!/usr/bin/env python3
"""
MLX Distributed Node using Exo's approach
Each node runs independently with its own shard
Based on exo's working implementation
"""
import argparse
import asyncio
import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

import grpc
from concurrent import futures
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.sample_utils import make_sampler

# Copy Shard class to avoid exo dependencies
from dataclasses import dataclass

@dataclass(frozen=True)
class Shard:
    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int
    
    def is_first_layer(self) -> bool:
        return self.start_layer == 0
    
    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers - 1
    
    def contains_layer(self, layer_idx: int) -> bool:
        return self.start_layer <= layer_idx <= self.end_layer

# We'll use our own simplified loader to avoid exo dependencies

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU as default
mx.set_default_device(mx.gpu)


class ShardedNode:
    """A node that handles one shard of the model"""
    
    def __init__(self, node_id: str, shard: Shard, next_node_address: Optional[str] = None):
        self.node_id = node_id
        self.shard = shard
        self.next_node_address = next_node_address
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.cache_states = {}  # Request ID -> cache state
        
    def load_model(self):
        """Load the sharded model using exo's approach"""
        logger.info(f"Node {self.node_id}: Loading shard layers {self.shard.start_layer}-{self.shard.end_layer}")
        
        # Get model path
        model_name = self.shard.model_id
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_id = f"models--{model_name.replace('/', '--')}"
        model_path = cache_dir / model_id / "snapshots"
        
        if model_path.exists():
            model_path = sorted([d for d in model_path.iterdir() if d.is_dir()])[-1]
            self.model_path = model_path
        else:
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Use our sharded loader
        from sharded_model_loader import load_sharded_model
        self.model = load_sharded_model(str(self.model_path), self.shard, lazy=False)
        
        # Load tokenizer if needed (first or last shard)
        if self.shard.is_first_layer() or self.shard.is_last_layer():
            self.tokenizer = load_tokenizer(self.model_path)
        
        memory_gb = mx.get_active_memory() / 1e9
        logger.info(f"Node {self.node_id}: Shard loaded, memory: {memory_gb:.2f} GB")
    
    def process_tensor(self, input_data: np.ndarray, request_id: str, use_cache: bool = True) -> np.ndarray:
        """Process tensor through this shard"""
        # Convert to MLX array
        x = mx.array(input_data)
        
        # Get or create cache for this request
        cache = None
        if use_cache and hasattr(self.model, 'layers'):
            if request_id not in self.cache_states:
                # Create cache for each layer in this shard
                from mlx_lm.models.cache import RotatingKVCache
                cache = []
                for layer in self.model.layers:
                    if hasattr(layer, 'self_attn'):
                        # This is a real transformer layer
                        cache.append(RotatingKVCache(
                            k_heads=layer.self_attn.n_kv_heads,
                            v_heads=layer.self_attn.n_kv_heads,
                            head_dim=layer.self_attn.head_dim,
                            max_size=512
                        ))
                    else:
                        # This is an IdentityBlock
                        cache.append(None)
                self.cache_states[request_id] = cache
            else:
                cache = self.cache_states[request_id]
        
        # Process through model
        output = self.model(x, cache)
        mx.eval(output)
        
        # Convert back to numpy
        return np.array(output)
    
    def encode_prompt(self, prompt: str) -> np.ndarray:
        """Encode prompt to tokens (only for first shard)"""
        if not self.shard.is_first_layer():
            raise ValueError("Only first shard can encode prompts")
        
        tokens = self.tokenizer.encode(prompt)
        return np.array(tokens)
    
    def decode_tokens(self, tokens: np.ndarray) -> str:
        """Decode tokens to text (only for last shard)"""
        if not self.shard.is_last_layer():
            raise ValueError("Only last shard can decode tokens")
        
        return self.tokenizer.decode(tokens.tolist())
    
    def sample_token(self, logits: np.ndarray, temperature: float = 0.7) -> int:
        """Sample next token from logits (only for last shard)"""
        if not self.shard.is_last_layer():
            raise ValueError("Only last shard can sample tokens")
        
        sampler = make_sampler(temp=temperature)
        logits_mx = mx.array(logits)
        logits_last = logits_mx[:, -1, :]
        token = sampler(logits_last)
        mx.eval(token)
        return int(token.item())


class GRPCNodeServer:
    """gRPC server for the node"""
    
    def __init__(self, node: ShardedNode, host: str, port: int):
        self.node = node
        self.host = host
        self.port = port
        self.server = None
        self.next_node_channel = None
        self.next_node_stub = None
        
        # Connect to next node if specified
        if self.node.next_node_address:
            self.connect_to_next_node()
    
    def connect_to_next_node(self):
        """Connect to the next node in the pipeline"""
        self.next_node_channel = grpc.insecure_channel(
            self.node.next_node_address,
            options=[
                ('grpc.max_send_message_length', 256 * 1024 * 1024),
                ('grpc.max_receive_message_length', 256 * 1024 * 1024),
            ]
        )
        # We'll use raw gRPC calls for simplicity
        logger.info(f"Connected to next node at {self.node.next_node_address}")
    
    async def start(self):
        """Start the gRPC server"""
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 256 * 1024 * 1024),
                ('grpc.max_receive_message_length', 256 * 1024 * 1024),
            ]
        )
        
        # For now, we'll use HTTP endpoints instead of gRPC for simplicity
        # This matches what exo does with their ChatGPT API
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI()
        
        class ProcessRequest(BaseModel):
            tensor_data: str  # Base64 encoded
            shape: list
            dtype: str
            request_id: str
            
        class ProcessResponse(BaseModel):
            tensor_data: str  # Base64 encoded
            shape: list
            dtype: str
            is_final: bool
        
        @app.post("/process")
        async def process(request: ProcessRequest):
            """Process tensor through this shard"""
            import base64
            
            # Decode input tensor
            tensor_bytes = base64.b64decode(request.tensor_data)
            input_array = np.frombuffer(tensor_bytes, dtype=request.dtype).reshape(request.shape)
            
            # Process through shard
            output_array = self.node.process_tensor(input_array, request.request_id)
            
            # If this is the last shard, we're done
            if self.node.shard.is_last_layer():
                return ProcessResponse(
                    tensor_data=base64.b64encode(output_array.tobytes()).decode(),
                    shape=list(output_array.shape),
                    dtype=str(output_array.dtype),
                    is_final=True
                )
            
            # Otherwise, forward to next node
            if self.node.next_node_address:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    next_request = {
                        "tensor_data": base64.b64encode(output_array.tobytes()).decode(),
                        "shape": list(output_array.shape),
                        "dtype": str(output_array.dtype),
                        "request_id": request.request_id
                    }
                    
                    async with session.post(
                        f"http://{self.node.next_node_address}/process",
                        json=next_request
                    ) as resp:
                        result = await resp.json()
                        return ProcessResponse(**result)
            
            # Should not reach here
            raise HTTPException(status_code=500, detail="Pipeline incomplete")
        
        @app.get("/health")
        async def health():
            return {
                "node_id": self.node.node_id,
                "shard": {
                    "start_layer": self.node.shard.start_layer,
                    "end_layer": self.node.shard.end_layer,
                    "n_layers": self.node.shard.n_layers
                },
                "memory_gb": mx.get_active_memory() / 1e9
            }
        
        # Start FastAPI server
        logger.info(f"Starting node {self.node.node_id} on {self.host}:{self.port}")
        await uvicorn.Server(
            uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        ).serve()


async def main():
    parser = argparse.ArgumentParser(description="MLX Distributed Node (Exo-style)")
    parser.add_argument("--node-id", required=True, help="Node ID")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=50051, help="Port")
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-8bit", help="Model name")
    parser.add_argument("--start-layer", type=int, required=True, help="Start layer")
    parser.add_argument("--end-layer", type=int, required=True, help="End layer")
    parser.add_argument("--n-layers", type=int, default=28, help="Total layers")
    parser.add_argument("--next-node", help="Next node address (e.g., 192.168.5.2:50051)")
    
    args = parser.parse_args()
    
    # Create shard
    shard = Shard(
        model_id=args.model,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        n_layers=args.n_layers
    )
    
    # Create node
    node = ShardedNode(
        node_id=args.node_id,
        shard=shard,
        next_node_address=args.next_node
    )
    
    # Load model
    node.load_model()
    
    # Start server
    server = GRPCNodeServer(node, args.host, args.port)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())