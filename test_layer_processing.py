#!/usr/bin/env python3
"""
Test individual layer processing to debug distributed inference.
"""

import mlx.core as mx
from mlx_lm import load
import numpy as np

# Load model
print("Loading model...")
model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")

# Test input
test_prompt = "What is machine learning?"
input_ids = mx.array(tokenizer.encode(test_prompt))
print(f"Input shape: {input_ids.shape}")

# Get embeddings
embeddings = model.model.embed_tokens(input_ids[None, :])
print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

# Test layer 10 directly
hidden_states = embeddings
for i in range(11):  # Process layers 0-10
    layer = model.model.layers[i]
    residual = hidden_states
    
    # Input layer norm
    hidden_states = layer.input_layernorm(hidden_states)
    
    # Self attention
    attn_output = layer.self_attn(hidden_states)
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]
    hidden_states = residual + attn_output
    residual = hidden_states
    
    # Post attention layer norm
    hidden_states = layer.post_attention_layernorm(hidden_states)
    
    # MLP
    mlp_output = layer.mlp(hidden_states)
    hidden_states = residual + mlp_output
    
    if i == 10:
        print(f"\nAfter layer {i}:")
        print(f"  Shape: {hidden_states.shape}")
        print(f"  Dtype: {hidden_states.dtype}")
        print(f"  Mean: {mx.mean(hidden_states).item()}")
        print(f"  First few: {hidden_states[0, 0, :5].tolist()}")

# Now let's test what happens when we simulate worker processing
print("\n\nSimulating worker processing...")

# Convert to numpy and back (simulating serialization)
np_array = np.array(hidden_states, copy=True)
print(f"Numpy dtype: {np_array.dtype}")

# Convert back to MLX
reconstructed = mx.array(np_array)
print(f"Reconstructed dtype: {reconstructed.dtype}")

# Process through one more layer with reconstructed tensor
layer_11 = model.model.layers[11]
residual = reconstructed

hidden_states_2 = layer_11.input_layernorm(residual)
attn_output_2 = layer_11.self_attn(hidden_states_2)
if isinstance(attn_output_2, tuple):
    attn_output_2 = attn_output_2[0]
hidden_states_2 = residual + attn_output_2
residual_2 = hidden_states_2

hidden_states_2 = layer_11.post_attention_layernorm(hidden_states_2)
mlp_output_2 = layer_11.mlp(hidden_states_2)
hidden_states_2 = residual_2 + mlp_output_2

print(f"\nAfter layer 11 (reconstructed input):")
print(f"  Shape: {hidden_states_2.shape}")
print(f"  Dtype: {hidden_states_2.dtype}")
print(f"  Mean: {mx.mean(hidden_states_2).item()}")
print(f"  First few: {hidden_states_2[0, 0, :5].tolist()}")

# Compare with direct processing
hidden_states_direct = model.model.layers[11](hidden_states)
if isinstance(hidden_states_direct, tuple):
    hidden_states_direct = hidden_states_direct[0]

print(f"\nDirect layer 11 processing:")
print(f"  Mean: {mx.mean(hidden_states_direct).item()}")
print(f"  First few: {hidden_states_direct[0, 0, :5].tolist()}")

# Calculate difference
diff = mx.abs(hidden_states_2 - hidden_states_direct)
print(f"\nDifference:")
print(f"  Max: {mx.max(diff).item()}")
print(f"  Mean: {mx.mean(diff).item()}")