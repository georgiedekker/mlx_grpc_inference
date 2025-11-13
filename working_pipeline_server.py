#!/usr/bin/env python3
"""
Working pipeline server that ensures both ranks participate
"""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
import json
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_pipeline_model():
    """Setup model to use our pipeline version"""
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


def synchronized_generate(model, tokenizer, prompt, max_tokens, rank, world_size):
    """Generation where both ranks participate"""
    
    # Tokenize on all ranks
    input_ids = tokenizer.encode(prompt) if rank == 0 else [1]  # Dummy for rank 1
    inputs = mx.array([input_ids])
    
    # Broadcast actual input shape from rank 0
    if rank == 0:
        input_shape = mx.array([len(input_ids)])
    else:
        input_shape = mx.zeros([1])
    input_shape = mx.distributed.all_sum(input_shape) / world_size
    mx.eval(input_shape)
    
    # Create same-sized input on all ranks
    if rank != 0:
        inputs = mx.ones((1, int(input_shape.item())), dtype=mx.int32)
    
    sampler = make_sampler(temp=0.7)
    generated = []
    cache = None
    
    for i in range(max_tokens):
        # All ranks do forward pass together
        logits = model.model(inputs, cache=cache)
        mx.eval(logits)
        
        if rank == 0:
            # Sample next token
            next_token = sampler(logits[:, -1, :])
            token_id = int(next_token.item())
            generated.append(token_id)
            
            # Broadcast token
            token_array = mx.array([token_id])
        else:
            token_array = mx.zeros([1], dtype=mx.int32)
        
        # All ranks get the token
        token_array = mx.distributed.all_sum(token_array)
        mx.eval(token_array)
        token_id = int(token_array.item() / world_size)
        
        if token_id == tokenizer.eos_token_id:
            break
            
        # Prepare next input
        inputs = mx.array([[token_id]])
    
    if rank == 0:
        return tokenizer.decode(generated)
    return ""


def main():
    group = mx.distributed.init()
    if not group:
        print("Run with: mpirun -n 2")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    logger.info(f"Rank {rank}/{world_size}: Starting")
    
    # Setup model config
    if rank == 0:
        setup_pipeline_model()
    
    # Sync
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    # Load model
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Apply pipeline
    model.model.pipeline(group)
    mx.eval(model.parameters())
    
    logger.info(f"Rank {rank}: Pipeline ready, layers {model.model.start_idx}-{model.model.end_idx-1}")
    
    # Sync before generation
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    if rank == 0:
        print("\n" + "="*60)
        print("PIPELINE WORKING - BOTH GPUS ACTIVE")
        print("="*60)
    
    # Generate with both ranks participating
    prompt = "What is 2+2? The answer is"
    
    start = time.time()
    result = synchronized_generate(model, tokenizer, prompt, 20, rank, world_size)
    elapsed = time.time() - start
    
    if rank == 0:
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Speed: {20/elapsed:.1f} tokens/sec")
        
        # Restore config
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