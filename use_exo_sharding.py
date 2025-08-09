#\!/usr/bin/env python3
"""
Use exo's existing sharding implementation for Qwen3
"""

import sys
import os

# Add exo to path
sys.path.insert(0, '/Users/mini1/Movies/exo')

from pathlib import Path
import mlx.core as mx
from exo.inference.shard import Shard
from exo.inference.mlx.sharded_utils import load_model_shard
from mlx_lm import load, stream_generate

def main():
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("No distributed group. Run with mpirun -n 2")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    print(f"Rank {rank}/{world_size}: Using exo sharding")
    
    # Download/cache the model first
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Get model path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_id = "models--mlx-community--Qwen3-1.7B-8bit"
    model_path = cache_dir / model_id
    
    # Find snapshot directory
    snapshots = model_path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
        if snapshot_dirs:
            model_path = snapshot_dirs[-1]
    
    print(f"Model path: {model_path}")
    
    # Define shards
    TOTAL_LAYERS = 28
    layers_per_device = TOTAL_LAYERS // world_size
    
    if rank == 0:
        # Rank 0 gets last layers
        shard = Shard(
            model_id="mlx-community/Qwen3-1.7B-8bit",
            start_layer=layers_per_device,
            end_layer=TOTAL_LAYERS - 1,
            n_layers=TOTAL_LAYERS
        )
    else:
        # Rank 1 gets first layers
        shard = Shard(
            model_id="mlx-community/Qwen3-1.7B-8bit",
            start_layer=0,
            end_layer=layers_per_device - 1,
            n_layers=TOTAL_LAYERS
        )
    
    print(f"Rank {rank}: Shard layers {shard.start_layer}-{shard.end_layer}")
    
    # Check if exo has Qwen3 support
    try:
        # Try to load with exo's sharded loader
        sharded_model = load_model_shard(model_path, shard, lazy=False)
        print(f"Rank {rank}: Successfully loaded shard\!")
        
        # Test forward pass
        test_input = mx.array([[1, 2, 3, 4, 5]])
        output = sharded_model(test_input)
        mx.eval(output)
        print(f"Rank {rank}: Forward pass successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"Rank {rank}: Failed to load with exo: {e}")
        print("Note: exo may need Qwen3 support added (currently has Qwen2)")


if __name__ == "__main__":
    main()