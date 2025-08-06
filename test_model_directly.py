#!/usr/bin/env python3
"""Test the model directly with MLX generate."""
import mlx.core as mx
from mlx_lm import load, generate

def test_direct():
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Test with simple generation
    prompt = "Hello"
    print(f"Generating from prompt: {prompt}")
    
    result = generate(model, tokenizer, prompt=prompt, max_tokens=20)
    print(f"Result: {result}")
    
    # Now test with chat template
    messages = [{"role": "user", "content": "Hello"}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nChat prompt: {chat_prompt}")
    
    result2 = generate(model, tokenizer, prompt=chat_prompt, max_tokens=20)
    print(f"Chat result: {result2}")

if __name__ == "__main__":
    test_direct()