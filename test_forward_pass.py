#!/usr/bin/env python3
"""
Test if distributed forward pass matches single device.
"""

import mlx.core as mx
from mlx_lm import load
import numpy as np

# Load model
print("Loading model...")
model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")

# Test input
test_prompt = "Hello world"
input_ids = mx.array(tokenizer.encode(test_prompt))
if len(input_ids.shape) == 1:
    input_ids = input_ids[None, :]  # Add batch dimension

print(f"Input shape: {input_ids.shape}")
print(f"Input tokens: {input_ids.tolist()}")

# Single device forward pass
print("\n=== Single Device Forward Pass ===")
embeddings = model.model.embed_tokens(input_ids)
hidden_states = embeddings

# Process all layers
for i in range(len(model.model.layers)):
    layer_output = model.model.layers[i](hidden_states)
    if isinstance(layer_output, tuple):
        hidden_states = layer_output[0]
    else:
        hidden_states = layer_output
    
    if i in [9, 18, 27]:  # After each device's layers
        print(f"After layer {i}: shape={hidden_states.shape}, mean={mx.mean(hidden_states).item():.4f}")

# Final norm
hidden_states = model.model.norm(hidden_states)
print(f"After final norm: shape={hidden_states.shape}, mean={mx.mean(hidden_states).item():.4f}")

# Get logits
logits = model.model.embed_tokens.as_linear(hidden_states)
print(f"Logits shape: {logits.shape}")

# Get top 5 predictions
logits_last = logits[0, -1, :]
sorted_indices = mx.argsort(logits_last)[::-1]  # Descending order
print("\nTop 5 predictions:")
for i in range(5):
    token_id = sorted_indices[i].item()
    logit_value = logits_last[token_id].item()
    token = tokenizer.decode([token_id])
    print(f"  {token_id}: '{token}' (logit={logit_value:.4f})")