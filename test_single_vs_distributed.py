#!/usr/bin/env python3
"""Compare single-device vs distributed inference outputs."""
import os
import asyncio
import mlx.core as mx
from mlx_lm import load, generate
import numpy as np

async def test_comparison():
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Test prompt
    messages = [{"role": "user", "content": "Hello!"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: {prompt}")
    print(f"Tokens: {tokens}")
    
    # Test 1: Single-device inference
    print("\n=== SINGLE-DEVICE TEST ===")
    result = generate(model, tokenizer, prompt=prompt, max_tokens=20)
    print(f"Single-device result: {result}")
    
    # Test 2: Just process through layers without generation
    print("\n=== LAYER-BY-LAYER TEST ===")
    input_ids = mx.array([tokens])
    
    with mx.stream(mx.gpu):
        # Embeddings
        hidden = model.model.embed_tokens(input_ids)
        mx.eval(hidden)
        print(f"After embeddings - std: {hidden.std():.2f}")
        
        # Process layers 0-13 (what mini1 would do)
        for i in range(0, 14):
            hidden = model.model.layers[i](hidden)
            mx.eval(hidden)
        print(f"After layers 0-13 - std: {hidden.std():.2f}, mean: {hidden.mean():.4f}")
        
        # Process layers 14-27 (what mini2 would do)
        for i in range(14, 28):
            hidden = model.model.layers[i](hidden)
            mx.eval(hidden)
        print(f"After layers 14-27 - std: {hidden.std():.2f}, mean: {hidden.mean():.4f}")
        
        # Final projection
        hidden = model.model.norm(hidden)
        if hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'as_linear'):
            logits = model.model.embed_tokens.as_linear(hidden)
        mx.eval(logits)
        print(f"Final logits - std: {logits.std():.2f}")
        
        # Check top tokens
        last_logits = logits[0, -1, :]
        top_indices = mx.argpartition(-last_logits, 5)[:5]
        top_values = last_logits[top_indices]
        sorted_idx = mx.argsort(-top_values)
        top_indices = top_indices[sorted_idx]
        
        top_tokens = [tokenizer.decode([int(idx)]) for idx in top_indices.tolist()]
        print(f"Top 5 predicted tokens: {top_tokens}")

if __name__ == "__main__":
    asyncio.run(test_comparison())