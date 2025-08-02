#!/usr/bin/env python3
"""
Run the single-device OpenAI-compatible API server.
For distributed inference, use run_distributed_openai.py instead.
"""

import uvicorn
import sys

print("WARNING: This runs the single-device API server.")
print("For distributed inference, use: uv run python run_distributed_openai.py")
print()

if __name__ == "__main__":
    uvicorn.run("openai_api:app", host="0.0.0.0", port=8100, reload=True)