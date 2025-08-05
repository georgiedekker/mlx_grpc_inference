#!/usr/bin/env python3
"""
Test inference using only local layers (first 10 layers on coordinator).
"""
import requests
import json

# Test with a simple prompt
url = "http://localhost:8100/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "messages": [
        {"role": "user", "content": "Hi"}
    ],
    "max_tokens": 5,  # Very short to test
    "temperature": 0.7
}

print("Testing local inference...")
print(f"Request: {json.dumps(data, indent=2)}")
print()

try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")