#!/usr/bin/env python3
"""Test how to properly call the model."""
from mlx_lm import load
import mlx.core as mx

model_name = "mlx-community/Qwen3-1.7B-8bit"
print(f"Loading {model_name}...")
model, tokenizer = load(model_name, lazy=True)

# Test input
test_text = "Hello"
input_ids = mx.array(tokenizer.encode(test_text)).reshape(1, -1)
print(f"\nInput shape: {input_ids.shape}")

# Method 1: Direct model call
print("\n=== Method 1: Direct model call ===")
try:
    output = model(input_ids)
    print(f"✓ Success! Output shape: {output.shape}")
    print(f"Output type: {type(output)}")
except Exception as e:
    print(f"✗ Error: {e}")

# Method 2: Through model.model
print("\n=== Method 2: Through model.model ===")
try:
    # Get embeddings
    embeddings = model.model.embed_tokens(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Pass through layers
    hidden = embeddings
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden)
        if i == 0:  # Just check first layer
            print(f"After layer 0: {hidden.shape}")
    
    # Apply final norm
    hidden = model.model.norm(hidden)
    print(f"After norm: {hidden.shape}")
    
    # Apply output projection
    output = model.model.embed_tokens.as_linear(hidden)
    print(f"✓ Final output shape: {output.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()