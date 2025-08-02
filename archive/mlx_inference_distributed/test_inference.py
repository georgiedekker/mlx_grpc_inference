#!/usr/bin/env python3
"""Test the distributed inference API."""

import requests
import json

url = "http://localhost:8100/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "model": "mlx-community/Qwen3-1.7B-8bit",
    "messages": [
        {
            "role": "user",
            "content": "Hello! What is distributed computing?"
        }
    ],
    "max_tokens": 50,
    "temperature": 0.7
}

print("Testing 3-device distributed inference...")
print(f"Request: {json.dumps(data, indent=2)}")

try:
    response = requests.post(url, headers=headers, json=data)
    print(f"\nStatus code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse: {json.dumps(result, indent=2)}")
        
        # Extract the assistant's response
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]["content"]
            print(f"\nAssistant: {message}")
    else:
        print(f"\nError: {response.text}")
        
except Exception as e:
    print(f"\nException: {e}")