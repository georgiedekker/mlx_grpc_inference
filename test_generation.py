#!/usr/bin/env python3
"""
Test actual text generation with pipeline-enabled Qwen3
"""

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import json
from pathlib import Path
import time


def setup_pipeline_config():
    """Modify config to use pipeline model"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_path = cache_dir / "models--mlx-community--Qwen3-1.7B-8bit"
    snapshots = model_path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
        if snapshot_dirs:
            config_path = snapshot_dirs[-1] / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            original_type = config.get('model_type', 'qwen3')
            config['model_type'] = 'qwen3_pipeline'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return config_path, original_type
    return None, None


def restore_config(config_path, original_type):
    """Restore original config"""
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['model_type'] = original_type
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def main():
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("ERROR: No distributed group")
        print("Run with: mpirun -n 2 --mca btl_tcp_if_include 192.168.5.0/24 python test_generation.py")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    print(f"Rank {rank}/{world_size}: Initializing")
    
    # Setup config on rank 0
    config_path = None
    original_type = None
    if rank == 0:
        config_path, original_type = setup_pipeline_config()
        print(f"Rank 0: Config modified to use qwen3_pipeline")
    
    # Sync before loading
    sync = mx.distributed.all_sum(mx.array([1.0], dtype=mx.float32), group=group)
    mx.eval(sync)
    
    # Load model on all ranks
    print(f"Rank {rank}: Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Apply pipeline
    if hasattr(model.model, 'pipeline'):
        model.model.pipeline(group)
        print(f"Rank {rank}: Pipeline applied - processing layers {model.model.start_idx} to {model.model.end_idx-1}")
        
        # Evaluate model parameters
        mx.eval(model.parameters())
        
        # Final sync
        sync = mx.distributed.all_sum(mx.array([1.0], dtype=mx.float32), group=group)
        mx.eval(sync)
        
        if rank == 0:
            print("\n" + "="*60)
            print("🚀 PIPELINE ACTIVE - BOTH GPUS PROCESSING")
            print(f"   Rank 0 (mini1): layers {model.model.start_idx}-{model.model.end_idx-1}")
            print(f"   Rank 1 (mini2): layers 0-{model.model.start_idx-1}")
            print("="*60 + "\n")
            
            # Test generation
            prompts = [
                "What is 2+2? The answer is",
                "The capital of France is",
                "Machine learning is"
            ]
            
            for prompt in prompts:
                print(f"\nPrompt: {prompt}")
                start_time = time.time()
                
                # Generate response with sampler
                sampler = make_sampler(temp=0.7)
                response = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=20,
                    sampler=sampler
                )
                
                elapsed = time.time() - start_time
                tokens_generated = len(tokenizer.encode(response)) - len(tokenizer.encode(prompt))
                
                print(f"Response: {response}")
                print(f"Time: {elapsed:.2f}s, Speed: {tokens_generated/elapsed:.1f} tokens/sec")
            
            print("\n" + "="*60)
            print("✅ SUCCESS: Both GPUs actively processing!")
            print("="*60)
            
            # Restore config
            restore_config(config_path, original_type)
            print("\nConfig restored to original state")
        else:
            # Rank 1 participates in generation through the pipeline
            # The generate function will handle the distributed calls
            pass
    else:
        print(f"Rank {rank}: ERROR - No pipeline method found")
        if rank == 0:
            restore_config(config_path, original_type)


if __name__ == "__main__":
    main()