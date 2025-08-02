#!/usr/bin/env python3
import requests
import json
import sys

# Quick test to see if OpenAI API is running
try:
    response = requests.get("http://localhost:8100/v1/models")
    if response.status_code == 200:
        print("✅ OpenAI-compatible API is running!")
        print("Models available:", json.dumps(response.json(), indent=2))
    else:
        print("❌ Server is running but not the OpenAI-compatible API")
        print("Please stop the current server and run:")
        print("  uv run python run_openai_server.py")
except requests.exceptions.ConnectionError:
    print("❌ No server is running on port 8100")
    print("Please run:")
    print("  uv run python run_openai_server.py")
    sys.exit(1)