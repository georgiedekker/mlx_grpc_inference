#!/usr/bin/env python3
"""Test the fix - pass mask and cache to layers."""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

def test_with_mask():
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Test prompt
    messages = [{"role": "user", "content": "Hello!"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt)
    print(f"Testing with {len(tokens)} tokens")
    
    input_ids = mx.array([tokens])
    
    # Test 1: WITHOUT mask (broken)
    print("\n=== WITHOUT MASK (BROKEN) ===")
    with mx.stream(mx.gpu):
        hidden = model.model.embed_tokens(input_ids)
        mx.eval(hidden)
        
        for i in range(28):
            hidden = model.model.layers[i](hidden)  # NO MASK!
            mx.eval(hidden)
        
        hidden = model.model.norm(hidden)
        logits = model.model.embed_tokens.as_linear(hidden)
        mx.eval(logits)
        
        top_token_id = mx.argmax(logits[0, -1, :]).item()
        top_token = tokenizer.decode([int(top_token_id)])
        print(f"Top predicted token: '{top_token}'")
    
    # Test 2: WITH mask (correct)
    print("\n=== WITH MASK (CORRECT) ===")
    with mx.stream(mx.gpu):
        hidden = model.model.embed_tokens(input_ids)
        mx.eval(hidden)
        
        # CREATE ATTENTION MASK
        mask = create_attention_mask(hidden, cache=None)
        cache = [None] * len(model.model.layers)
        
        for i, c in zip(range(28), cache):
            hidden = model.model.layers[i](hidden, mask, c)  # WITH MASK!
            mx.eval(hidden)
        
        hidden = model.model.norm(hidden)
        logits = model.model.embed_tokens.as_linear(hidden)
        mx.eval(logits)
        
        top_token_id = mx.argmax(logits[0, -1, :]).item()
        top_token = tokenizer.decode([int(top_token_id)])
        print(f"Top predicted token: '{top_token}'")
    
    # Test 3: Split with mask
    print("\n=== SPLIT WITH MASK (SIMULATING DISTRIBUTED) ===")
    with mx.stream(mx.gpu):
        hidden = model.model.embed_tokens(input_ids)
        mx.eval(hidden)
        
        # CREATE ATTENTION MASK
        mask = create_attention_mask(hidden, cache=None)
        cache = [None] * len(model.model.layers)
        
        # Stage 1: Layers 0-13
        for i in range(14):
            hidden = model.model.layers[i](hidden, mask, cache[i])
            mx.eval(hidden)
        mx.synchronize()
        print(f"After layers 0-13: std={hidden.std():.2f}")
        
        # Simulate transfer
        if hidden.dtype == mx.bfloat16:
            hidden = hidden.astype(mx.float32)
        import numpy as np
        hidden_numpy = np.array(hidden, copy=True)
        hidden = mx.array(hidden_numpy)
        mx.eval(hidden)
        
        # Stage 2: Layers 14-27
        for i in range(14, 28):
            hidden = model.model.layers[i](hidden, mask, cache[i])
            mx.eval(hidden)
        mx.synchronize()
        print(f"After layers 14-27: std={hidden.std():.2f}")
        
        hidden = model.model.norm(hidden)
        logits = model.model.embed_tokens.as_linear(hidden)
        mx.eval(logits)
        
        top_token_id = mx.argmax(logits[0, -1, :]).item()
        top_token = tokenizer.decode([int(top_token_id)])
        print(f"Top predicted token: '{top_token}'")

if __name__ == "__main__":
    test_with_mask()