#!/usr/bin/env python3
"""
Debug the pipeline forward pass
"""

import mlx.core as mx
from mlx_lm import load
import json
from pathlib import Path


def setup_model():
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_path = cache_dir / "models--mlx-community--Qwen3-1.7B-8bit"
    snapshots = model_path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
        if snapshot_dirs:
            config_path = snapshot_dirs[-1] / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            config['model_type'] = 'qwen3_pipeline'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)


def main():
    group = mx.distributed.init()
    rank = group.rank()
    world_size = group.size()
    
    print(f"Rank {rank}: Starting debug")
    
    if rank == 0:
        setup_model()
    
    # Sync
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    # Load model
    print(f"Rank {rank}: Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Apply pipeline
    model.model.pipeline(group)
    print(f"Rank {rank}: Pipeline applied - layers {model.model.start_idx}-{model.model.end_idx-1}")
    
    # Create test input
    test_input = mx.array([[1, 2, 3, 4, 5]])
    
    print(f"Rank {rank}: Calling forward pass...")
    
    # Both ranks call forward
    try:
        # The issue is here - in the pipelined forward
        # Rank 1 (first layers) needs to process first
        # Rank 0 (last layers) needs to wait for rank 1
        
        if rank == 1:
            print(f"Rank 1: Processing layers {model.model.start_idx}-{model.model.end_idx-1}")
            # Rank 1 should process and send
            output = model.model(test_input)
            print(f"Rank 1: Forward complete")
        else:
            print(f"Rank 0: Waiting for rank 1...")
            # Rank 0 should receive and process
            output = model.model(test_input)
            print(f"Rank 0: Forward complete")
            
        mx.eval(output)
        print(f"Rank {rank}: Success! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Rank {rank}: Error: {e}")
    
    # Cleanup
    if rank == 0:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_path = cache_dir / "models--mlx-community--Qwen3-1.7B-8bit"
        snapshots = model_path / "snapshots"
        if snapshots.exists():
            snapshot_dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
            if snapshot_dirs:
                config_path = snapshot_dirs[-1] / "config.json"
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config['model_type'] = 'qwen3'
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()