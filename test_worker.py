#!/usr/bin/env python3
"""Test a worker's model independently."""
import mlx.core as mx
from mlx_lm import load
import numpy as np

def test_model():
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Test input - use the same prompt as the server
    test_text = "Hello"
    # Apply chat template like the server does
    messages = [{"role": "user", "content": test_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    
    print(f"Testing with input: {test_text}")
    print(f"Tokens: {tokens}")
    
    # Process through model
    with mx.stream(mx.gpu):
        # Embeddings
        hidden = model.model.embed_tokens(input_ids)
        mx.eval(hidden)
        print(f"After embeddings - shape: {hidden.shape}, mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")
        
        # Process each layer
        for i, layer in enumerate(model.model.layers):
            hidden = layer(hidden)
            mx.eval(hidden)
            
            # Check for corruption
            if hidden.std() > 50:
                print(f"ERROR: Layer {i} produced corrupted output! std: {hidden.std():.6f}")
                print(f"This layer is broken!")
                return
            
            if i % 5 == 0:
                print(f"After layer {i} - mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")
        
        # Final norm
        hidden = model.model.norm(hidden)
        mx.eval(hidden)
        print(f"After final norm - mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")
        
        # Projection
        if hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'as_linear'):
            logits = model.model.embed_tokens.as_linear(hidden)
        mx.eval(logits)
        print(f"Final logits - shape: {logits.shape}, mean: {logits.mean():.6f}, std: {logits.std():.6f}")
        
        # Check top tokens
        last_logits = logits[0, -1, :]
        top_indices = mx.argmax(last_logits)
        top_token = tokenizer.decode([int(top_indices.item())])
        print(f"Top predicted token: {top_token}")
        
    print("\nModel test complete!")

if __name__ == "__main__":
    test_model()