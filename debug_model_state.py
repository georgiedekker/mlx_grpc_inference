#!/usr/bin/env python3
"""Debug why splitting layer processing breaks the model."""
import mlx.core as mx
from mlx_lm import load
import numpy as np

def test_continuous_vs_split():
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Test prompt
    messages = [{"role": "user", "content": "Hello!"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt)
    print(f"Testing with {len(tokens)} tokens")
    
    input_ids = mx.array([tokens])
    
    # Test 1: Continuous processing (how MLX generate does it)
    print("\n=== TEST 1: CONTINUOUS PROCESSING ===")
    with mx.stream(mx.gpu):
        hidden = model.model.embed_tokens(input_ids)
        mx.eval(hidden)
        print(f"Embeddings: std={hidden.std():.2f}")
        
        # Process ALL layers continuously
        for i in range(28):
            hidden = model.model.layers[i](hidden)
            mx.eval(hidden)
            if i == 13:
                print(f"After layer 13: std={hidden.std():.2f}, mean={hidden.mean():.4f}")
            elif i == 27:
                print(f"After layer 27: std={hidden.std():.2f}, mean={hidden.mean():.4f}")
        
        # Final projection
        hidden = model.model.norm(hidden)
        logits = model.model.embed_tokens.as_linear(hidden)
        mx.eval(logits)
        
        # Get top token
        top_token_id = mx.argmax(logits[0, -1, :]).item()
        top_token = tokenizer.decode([int(top_token_id)])
        print(f"Top predicted token: '{top_token}'")
    
    # Test 2: Split processing (what distributed does)
    print("\n=== TEST 2: SPLIT PROCESSING (SIMULATING DISTRIBUTED) ===")
    with mx.stream(mx.gpu):
        # Stage 1: Embeddings + layers 0-13
        hidden = model.model.embed_tokens(input_ids)
        mx.eval(hidden)
        mx.synchronize()  # Ensure complete
        print(f"Embeddings: std={hidden.std():.2f}")
        
        for i in range(14):
            hidden = model.model.layers[i](hidden)
            mx.eval(hidden)
        mx.synchronize()  # Critical synchronization point
        print(f"After layers 0-13: std={hidden.std():.2f}, mean={hidden.mean():.4f}")
        
        # SIMULATE SERIALIZATION/DESERIALIZATION
        # This is what happens when we send between devices
        # Handle potential dtype issues
        if hidden.dtype == mx.bfloat16:
            hidden = hidden.astype(mx.float32)
        hidden_numpy = np.array(hidden, copy=True)
        hidden = mx.array(hidden_numpy)
        mx.eval(hidden)
        mx.synchronize()
        print(f"After serialize/deserialize: std={hidden.std():.2f}, mean={hidden.mean():.4f}")
        
        # Stage 2: Layers 14-27
        for i in range(14, 28):
            hidden = model.model.layers[i](hidden)
            mx.eval(hidden)
        mx.synchronize()
        print(f"After layers 14-27: std={hidden.std():.2f}, mean={hidden.mean():.4f}")
        
        # Final projection
        hidden = model.model.norm(hidden)
        logits = model.model.embed_tokens.as_linear(hidden)
        mx.eval(logits)
        
        # Get top token
        top_token_id = mx.argmax(logits[0, -1, :]).item()
        top_token = tokenizer.decode([int(top_token_id)])
        print(f"Top predicted token: '{top_token}'")
    
    # Test 3: Check if the issue is with the specific layer split point
    print("\n=== TEST 3: DIFFERENT SPLIT POINTS ===")
    split_points = [7, 14, 21]
    for split in split_points:
        with mx.stream(mx.gpu):
            hidden = model.model.embed_tokens(input_ids)
            mx.eval(hidden)
            
            # First half
            for i in range(split):
                hidden = model.model.layers[i](hidden)
                mx.eval(hidden)
            mx.synchronize()
            
            # Simulate transfer
            if hidden.dtype == mx.bfloat16:
                hidden = hidden.astype(mx.float32)
            hidden_numpy = np.array(hidden, copy=True)
            hidden = mx.array(hidden_numpy)
            mx.eval(hidden)
            
            # Second half
            for i in range(split, 28):
                hidden = model.model.layers[i](hidden)
                mx.eval(hidden)
            mx.synchronize()
            
            # Final projection
            hidden = model.model.norm(hidden)
            logits = model.model.embed_tokens.as_linear(hidden)
            mx.eval(logits)
            
            top_token_id = mx.argmax(logits[0, -1, :]).item()
            top_token = tokenizer.decode([int(top_token_id)])
            print(f"Split at layer {split}: std={hidden.std():.2f}, top token='{top_token}'")

if __name__ == "__main__":
    test_continuous_vs_split()