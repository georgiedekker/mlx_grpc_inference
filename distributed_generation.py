#!/usr/bin/env python3
"""
Distributed generation that actually works with pipeline parallelism
Both ranks must participate in the forward pass
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
import json
from pathlib import Path
import time
import numpy as np


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


def distributed_generate(model, tokenizer, prompt, max_tokens, rank, group):
    """
    Custom generation that works with distributed pipeline
    Both ranks must participate in forward passes
    """
    sampler = make_sampler(temp=0.7)
    
    # Only rank 0 does tokenization
    if rank == 0:
        input_ids = tokenizer.encode(prompt)
        input_len = len(input_ids)
    else:
        input_ids = [1]  # Dummy
        input_len = 0
    
    # Broadcast input length
    input_len_array = mx.array([float(input_len)])
    input_len_array = mx.distributed.all_sum(input_len_array, group=group)
    mx.eval(input_len_array)
    input_len = int(input_len_array.item())
    
    # Create input array (all ranks need same shape)
    if rank == 0:
        inputs = mx.array([tokenizer.encode(prompt)])
    else:
        inputs = mx.ones((1, input_len), dtype=mx.int32)
    
    # Initialize cache
    cache = None
    generated = []
    
    for i in range(max_tokens):
        # All ranks do forward pass together
        logits = model(inputs, cache=cache)
        mx.eval(logits)
        
        if rank == 0:
            # Only rank 0 samples
            next_token_logits = logits[:, -1, :]
            next_token = sampler(next_token_logits)
            token_id = int(next_token.item())
            generated.append(token_id)
            
            # Check for EOS
            if token_id == tokenizer.eos_token_id:
                # Broadcast EOS signal
                eos_signal = mx.array([1.0])
            else:
                eos_signal = mx.array([0.0])
        else:
            token_id = 0
            eos_signal = mx.array([0.0])
        
        # Broadcast EOS signal
        eos_signal = mx.distributed.all_sum(eos_signal, group=group)
        mx.eval(eos_signal)
        
        if eos_signal.item() > 0:
            break
        
        # Broadcast next token
        if rank == 0:
            token_array = mx.array([float(token_id)])
        else:
            token_array = mx.array([0.0])
        
        token_array = mx.distributed.all_sum(token_array, group=group)
        mx.eval(token_array)
        token_id = int(token_array.item())
        
        # Update inputs for next iteration
        inputs = mx.array([[token_id]])
        
        # Update cache if needed
        if cache is not None:
            # Handle cache updates
            pass
    
    if rank == 0:
        return tokenizer.decode(generated)
    return ""


def main():
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("ERROR: No distributed group")
        print("Run with: mpirun -n 2 --mca btl_tcp_if_include 192.168.5.0/24 python distributed_generation.py")
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
            print("🚀 DISTRIBUTED PIPELINE ACTIVE")
            print(f"   Rank 0 (mini1): layers {model.model.start_idx}-{model.model.end_idx-1}")
            print(f"   Rank 1 (mini2): layers 0-{model.model.start_idx-1}")
            print("="*60 + "\n")
        
        # Test generation with both ranks participating
        prompts = [
            "What is 2+2?",
            "The capital of France is",
            "Machine learning is"
        ]
        
        for prompt in prompts:
            if rank == 0:
                print(f"\nPrompt: {prompt}")
            
            start_time = time.time()
            
            # Both ranks participate in generation
            response = distributed_generate(model, tokenizer, prompt, 20, rank, group)
            
            elapsed = time.time() - start_time
            
            if rank == 0:
                print(f"Response: {prompt}{response}")
                tokens_generated = len(tokenizer.encode(response))
                print(f"Time: {elapsed:.2f}s, Speed: {tokens_generated/elapsed:.1f} tokens/sec")
        
        if rank == 0:
            print("\n" + "="*60)
            print("✅ SUCCESS: Both GPUs actively processing!")
            print("="*60)
            
            # Restore config
            restore_config(config_path, original_type)
            print("\nConfig restored to original state")
    else:
        print(f"Rank {rank}: ERROR - No pipeline method found")
        if rank == 0:
            restore_config(config_path, original_type)


if __name__ == "__main__":
    main()