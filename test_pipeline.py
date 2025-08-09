#!/usr/bin/env python3
"""
Test Qwen3 with pipeline parallelism using the official MLX pattern
"""

import mlx.core as mx
from mlx_lm import load, stream_generate
from patch_qwen3 import add_pipeline_to_qwen3


def main():
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("No distributed group. Run with: mpirun -n 2 python test_pipeline.py")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    def rprint(*args, **kwargs):
        """Print only from rank 0"""
        if rank == 0:
            print(*args, **kwargs)
    
    rprint(f"Testing Qwen3 pipeline with {world_size} GPUs")
    
    # Load model
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Add pipeline support
    model = add_pipeline_to_qwen3(model)
    
    # Apply pipeline parallelism
    model.model.pipeline(group)
    
    # Evaluate parameters
    mx.eval(model.parameters())
    
    # Synchronize before generation
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    rprint("✅ Model loaded with pipeline support")
    rprint(f"   Rank 0: layers {14}-{27}")
    rprint(f"   Rank 1: layers {0}-{13}")
    
    # Test generation
    prompt = "What is 2+2? The answer is"
    rprint(f"\nPrompt: {prompt}")
    rprint("Response: ", end="")
    
    # Generate using stream_generate (which should work with pipeline)
    for response in stream_generate(
        model, 
        tokenizer, 
        prompt, 
        max_tokens=20
    ):
        rprint(response.text, end="", flush=True)
    
    rprint()
    
    if rank == 0:
        rprint("\n" + "="*50)
        rprint(f"Prompt tokens: {response.prompt_tokens}")
        rprint(f"Generation: {response.generation_tokens} tokens")
        rprint(f"Speed: {response.generation_tps:.1f} tokens/sec")
        rprint(f"Memory: {response.peak_memory:.2f} GB")


if __name__ == "__main__":
    main()