#!/usr/bin/env python3
"""
Simple test to verify distributed inference works.
"""

import mlx.core as mx
from mlx_lm import load, generate

# Test 1: Single device inference
print("=== Test 1: Single Device Inference ===")
model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")

test_prompts = ["Hi", "Hello", "What is 2+2?"]
for prompt in test_prompts:
    response = generate(model, tokenizer, prompt=prompt, max_tokens=20)
    print(f"Prompt: '{prompt}'")
    print(f"Response: {response}")
    print("-" * 40)