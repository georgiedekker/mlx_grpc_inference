#!/usr/bin/env python3
"""
Compare single device vs distributed forward pass.
This will help us identify where the computation diverges.
"""

import mlx.core as mx
from mlx_lm import load
import numpy as np

# Load model
print("Loading model...")  
model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")

# Test input - just "Hi"
test_prompt = "Hi"
input_ids = mx.array(tokenizer.encode(test_prompt))
if len(input_ids.shape) == 1:
    input_ids = input_ids[None, :]

print(f"Input: '{test_prompt}' -> tokens {input_ids.tolist()}")

# Get embeddings
embeddings = model.model.embed_tokens(input_ids)
print(f"Embeddings: shape={embeddings.shape}, mean={mx.mean(embeddings).item():.4f}")

# Process layers 0-9 (coordinator)
hidden_states = embeddings
for i in range(10):
    hidden_states = model.model.layers[i](hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
print(f"After layers 0-9: shape={hidden_states.shape}, mean={mx.mean(hidden_states).item():.4f}")

# Process layers 10-18 (mini2)
for i in range(10, 19):
    hidden_states = model.model.layers[i](hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
print(f"After layers 10-18: shape={hidden_states.shape}, mean={mx.mean(hidden_states).item():.4f}")

# Process layers 19-27 (m4)
for i in range(19, 28):
    hidden_states = model.model.layers[i](hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
print(f"After layers 19-27: shape={hidden_states.shape}, mean={mx.mean(hidden_states).item():.4f}")

# Apply norm
normed = model.model.norm(hidden_states)
print(f"After norm: shape={normed.shape}, mean={mx.mean(normed).item():.4f}")

# Get logits
logits = model.model.embed_tokens.as_linear(normed)
print(f"Logits: shape={logits.shape}")

# Check top prediction
top_token = mx.argmax(logits[0, -1, :]).item()
print(f"Top prediction: token {top_token} = '{tokenizer.decode([top_token])}'")

# Now let's do a proper generation to see what should happen
print("\n=== Proper generation ===")
from mlx_lm import generate
response = generate(model, tokenizer, prompt=test_prompt, max_tokens=5)
print(f"Generated: {response}")