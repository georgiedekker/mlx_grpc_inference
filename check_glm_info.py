#!/usr/bin/env python3
"""
Quick check of GLM model info from HuggingFace.
"""

import requests
import json

# Check model card
model_id = "mlx-community/GLM-4.5-Air-4bit"
api_url = f"https://huggingface.co/api/models/{model_id}"

print(f"Checking model: {model_id}")

try:
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        
        # Check tags for model info
        if 'tags' in data:
            print(f"\nTags: {data['tags'][:10]}")
        
        # Check files
        files_url = f"https://huggingface.co/api/models/{model_id}/tree/main"
        files_response = requests.get(files_url)
        if files_response.status_code == 200:
            files = files_response.json()
            print(f"\nModel files:")
            for file in files[:10]:
                if 'path' in file:
                    print(f"  - {file['path']} ({file.get('size', 0) / 1024 / 1024:.1f} MB)")
                    
        # Try to get config.json
        config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        config_response = requests.get(config_url)
        if config_response.status_code == 200:
            config = config_response.json()
            print(f"\nModel config:")
            print(f"  - Hidden size: {config.get('hidden_size', 'unknown')}")
            print(f"  - Number of layers: {config.get('num_hidden_layers', 'unknown')}")
            print(f"  - Number of attention heads: {config.get('num_attention_heads', 'unknown')}")
            print(f"  - Vocab size: {config.get('vocab_size', 'unknown')}")
            
except Exception as e:
    print(f"Error: {e}")
    
# Suggest alternative models
print("\n\nAlternative smaller models to consider:")
print("1. mlx-community/Qwen2.5-3B-4bit (3B parameters, 36 layers)")
print("2. mlx-community/Llama-3.2-3B-4bit (3B parameters, 28 layers)")
print("3. mlx-community/Mistral-7B-v0.3-4bit (7B parameters, 32 layers)")
print("4. mlx-community/Phi-4-4bit (14B parameters, 40 layers)")