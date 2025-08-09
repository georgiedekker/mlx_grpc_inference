#!/usr/bin/env python3
"""
Test official pipeline with DeepSeek model
Run with: mlx.launch --hostfile hosts.json --backend mpi test_pipeline_official.py
"""

import argparse
import mlx.core as mx
from mlx_lm import load, stream_generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        help="Model to use"
    )
    parser.add_argument(
        "--prompt",
        default="What is 2+2?",
        help="Prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens"
    )
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    size = group.size()
    
    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)
    
    rprint(f"Loading {args.model} on {size} devices...")
    
    # Load model normally
    model, tokenizer = load(args.model)
    
    # Check if model has pipeline support
    if hasattr(model, 'model') and hasattr(model.model, 'pipeline'):
        rprint("✅ Model has native pipeline() support!")
        model.model.pipeline(group)
        mx.eval(model.parameters())
    else:
        rprint("❌ Model does NOT have pipeline() support")
        rprint(f"Model type: {type(model).__name__}")
        if hasattr(model, 'model'):
            rprint(f"Inner model type: {type(model.model).__name__}")
        exit(1)
    
    # Synchronize
    mx.eval(mx.distributed.all_sum(mx.array(1.0)))
    
    # Generate
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    rprint("Generating...")
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens
    ):
        rprint(response.text, end="", flush=True)
    
    rprint()
    rprint("=" * 40)
    rprint(f"Prompt: {response.prompt_tokens} tokens, {response.prompt_tps:.1f} tok/s")
    rprint(f"Generation: {response.generation_tokens} tokens, {response.generation_tps:.1f} tok/s")
    rprint(f"Peak memory: {response.peak_memory:.2f} GB")