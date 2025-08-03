#!/usr/bin/env python3
"""
Minimal test to identify the exact issue with generation.
"""

from mlx_lm import load, generate
import mlx.core as mx

# Load model
print("Loading model...")
model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
print(f"Model loaded. Type: {type(model)}")

# Test 1: Simple generate without temperature
print("\n=== Test 1: generate() without temperature ===")
prompt = "Hello"
try:
    response = generate(model, tokenizer, prompt=prompt, max_tokens=20)
    print(f"Success! Response: '{response}'")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Simple generate with temperature parameter
print("\n=== Test 2: generate() with temperature ===")
try:
    response = generate(model, tokenizer, prompt=prompt, max_tokens=20, temperature=0.7)
    print(f"Success! Response: '{response}'")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Check actual generation step by step
print("\n=== Test 3: Manual generation ===")
from mlx_lm.sample_utils import make_sampler

# Tokenize
input_ids = mx.array(tokenizer.encode(prompt))
print(f"Input IDs: {input_ids.tolist()}")

# Add batch dimension
if len(input_ids.shape) == 1:
    input_ids = input_ids[None, :]

# Get embeddings
embeddings = model.model.embed_tokens(input_ids)
print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

# Process through first few layers
hidden_states = embeddings
for i, layer in enumerate(model.model.layers[:5]):
    layer_output = layer(hidden_states)
    if isinstance(layer_output, tuple):
        hidden_states = layer_output[0]
    else:
        hidden_states = layer_output
    print(f"After layer {i}: mean={mx.mean(hidden_states).item():.4f}, max={mx.max(hidden_states).item():.4f}")

# Check if values explode  
if mx.max(mx.abs(hidden_states)).item() > 1000:
    print("WARNING: Hidden states have exploded!")

# Process all layers
hidden_states = embeddings
for layer in model.model.layers:
    layer_output = layer(hidden_states)
    if isinstance(layer_output, tuple):
        hidden_states = layer_output[0]
    else:
        hidden_states = layer_output

# Norm and project
hidden_states = model.model.norm(hidden_states)
logits = model.model.embed_tokens.as_linear(hidden_states)
print(f"\nLogits shape: {logits.shape}, dtype: {logits.dtype}")
print(f"Logits stats: min={mx.min(logits).item():.4f}, max={mx.max(logits).item():.4f}")

# Check top predictions
last_logits = logits[0, -1, :]
top_indices = mx.argpartition(-last_logits, kth=5)[:5]
print("\nTop 5 predictions:")
for idx in top_indices:
    token_id = idx.item()
    logit_val = last_logits[token_id].item()
    token_text = tokenizer.decode([token_id])
    print(f"  Token {token_id} ('{token_text}'): {logit_val:.4f}")

# Sample one token
sampler = make_sampler(temp=0.7)
next_token = sampler(logits[:, -1:, :])
token_id = next_token.item()
token_text = tokenizer.decode([token_id])
print(f"\nSampled token: {token_id} ('{token_text}')")