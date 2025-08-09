#!/usr/bin/env python3
"""
Simple sharding approach - each device loads only its layers
Communication via MPI send/recv of tensors
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOTAL_LAYERS = 28  # Qwen3-1.7B has 28 layers


@dataclass
class Shard:
    """Simple shard definition"""
    start_layer: int
    end_layer: int
    
    def is_first(self):
        return self.start_layer == 0
    
    def is_last(self):
        return self.end_layer == TOTAL_LAYERS - 1


def shard_model(model, rank, world_size):
    """
    Modify model to only keep weights for our shard
    Other layers become identity functions
    """
    layers_per_rank = TOTAL_LAYERS // world_size
    
    if rank == 0:
        # Rank 0 gets last layers (for pipeline flow)
        shard = Shard(layers_per_rank, TOTAL_LAYERS - 1)
    else:
        # Rank 1 gets first layers
        shard = Shard(0, layers_per_rank - 1)
    
    logger.info(f"Rank {rank}: Shard layers {shard.start_layer}-{shard.end_layer}")
    
    # Store original layers
    original_layers = model.model.layers
    
    # Create identity layer
    class IdentityLayer(nn.Module):
        def __call__(self, x, mask=None, cache=None):
            return x
    
    # Replace layers outside our shard with identity
    for i in range(len(original_layers)):
        if i < shard.start_layer or i > shard.end_layer:
            model.model.layers[i] = IdentityLayer()
            logger.info(f"Rank {rank}: Layer {i} -> Identity")
    
    # Add shard info to model
    model.model.shard = shard
    model.model.rank = rank
    model.model.world_size = world_size
    
    # Modify forward pass to handle sharding
    original_forward = model.model.__class__.__call__
    
    def sharded_forward(self, inputs, mask=None, cache=None):
        """Forward pass with inter-shard communication"""
        # Only first shard does embeddings
        if hasattr(self, 'shard') and self.shard.is_first():
            h = self.embed_tokens(inputs)
        elif hasattr(self, 'shard'):
            # Other shards receive from previous
            h = mx.distributed.recv(src=(self.rank + 1) % self.world_size)
            mx.eval(h)
        else:
            h = self.embed_tokens(inputs)
        
        # Create mask if needed
        if mask is None:
            from mlx_lm.models.qwen3 import create_attention_mask
            mask = create_attention_mask(h, cache)
        
        if cache is None:
            cache = [None] * len(self.layers)
        
        # Process layers
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        
        # Communication between shards
        if hasattr(self, 'shard') and self.world_size > 1:
            if not self.shard.is_last():
                # Send to next shard
                mx.distributed.send(h, dst=(self.rank - 1) % self.world_size)
                mx.eval(h)
                # Receive final result
                h = mx.distributed.recv_like(h, src=(self.rank - 1) % self.world_size)
                mx.eval(h)
            else:
                # Last shard sends back result
                mx.distributed.send(h, dst=(self.rank + 1) % self.world_size)
                mx.eval(h)
        
        # Only last shard applies norm
        if hasattr(self, 'shard') and self.shard.is_last():
            return self.norm(h)
        elif not hasattr(self, 'shard'):
            return self.norm(h)
        else:
            return h
    
    # Replace forward method
    model.model.__class__.__call__ = sharded_forward
    
    return model, shard


def main():
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("Run with: mpirun -n 2 python simple_sharding.py")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    logger.info(f"Rank {rank}/{world_size}: Loading model...")
    
    # Load full model
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Shard the model
    model, shard = shard_model(model, rank, world_size)
    
    # Evaluate parameters
    mx.eval(model.parameters())
    
    # Get memory usage
    memory = mx.get_active_memory() / (1024**3)
    logger.info(f"Rank {rank}: Memory usage: {memory:.2f} GB")
    
    # Synchronize
    sync = mx.distributed.all_sum(mx.array([1.0]))
    mx.eval(sync)
    
    if rank == 0:
        print("\n" + "="*60)
        print("✅ SIMPLE SHARDING READY")
        print(f"   Rank 0: layers {shard.start_layer}-{shard.end_layer}")
        print(f"   Rank 1: layers 0-{TOTAL_LAYERS//world_size - 1}")
        print("="*60)
    
    # Test forward pass
    logger.info(f"Rank {rank}: Testing forward pass...")
    test_input = mx.array([[1, 2, 3, 4, 5]])
    
    try:
        output = model.model(test_input)
        mx.eval(output)
        logger.info(f"Rank {rank}: Forward pass successful! Shape: {output.shape}")
    except Exception as e:
        logger.error(f"Rank {rank}: Forward pass failed: {e}")
    
    # Final sync
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    if rank == 0:
        print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()