#!/usr/bin/env python3
"""
Final test of Qwen3 with pipeline support
"""

import mlx.core as mx
from mlx_lm import load, stream_generate
import json
from pathlib import Path


def main():
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("No distributed group. Run with: mpirun -n 2")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    print(f"Rank {rank}/{world_size}: Starting test")
    
    # Temporarily modify config to use qwen3_pipeline
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_path = cache_dir / "models--mlx-community--Qwen3-1.7B-8bit"
    snapshots = model_path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
        if snapshot_dirs:
            config_path = snapshot_dirs[-1] / "config.json"
            
            # Modify config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            original_type = config.get('model_type', 'qwen3')
            config['model_type'] = 'qwen3_pipeline'
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Rank {rank}: Modified config to use qwen3_pipeline")
    
    # Load model
    print(f"Rank {rank}: Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Check pipeline method
    if hasattr(model.model, 'pipeline'):
        print(f"Rank {rank}: ✅ Pipeline method found!")
        
        # Apply pipeline
        model.model.pipeline(group)
        print(f"Rank {rank}: Pipeline applied - layers {model.model.start_idx}-{model.model.end_idx-1}")
        
        # Evaluate
        mx.eval(model.parameters())
        
        # Synchronize
        sync = mx.distributed.all_sum(mx.array([1.0]))
        mx.eval(sync)
        
        if rank == 0:
            print("\n" + "="*60)
            print("✅ PIPELINE WORKING!")
            print(f"   Rank 0: layers {model.model.start_idx}-{model.model.end_idx-1}")
            print(f"   Rank 1: layers 0-{model.model.start_idx-1}")
            print("="*60)
            
            # Test generation
            print("\nTesting generation...")
            prompt = "What is 2+2?"
            
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=10
            ):
                print(response.text, end="", flush=True)
            
            print(f"\n\nSpeed: {response.generation_tps:.1f} tokens/sec")
            
            # Restore original config
            config['model_type'] = original_type
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("Config restored")
    else:
        print(f"Rank {rank}: ❌ No pipeline method")


if __name__ == "__main__":
    main()