#!/usr/bin/env python3
"""
Test local-only inference (single device).
"""
import mlx.core as mx
from mlx_lm import load, generate

# Load model
print("Loading model...")
model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")

# Test generation
prompt = "What is 2+2?"
print(f"\nPrompt: {prompt}")

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=20,
    temperature=0.7,
    verbose=True
)

print(f"\nResponse: {response}")