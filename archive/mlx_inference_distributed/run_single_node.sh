#!/bin/bash
# SINGLE NODE MLX INFERENCE - SIMPLE AND WORKING
# This runs everything on mini1 without distributed complexity

echo "🚀 SINGLE NODE MLX INFERENCE"
echo "============================"

cd /Users/mini1/Movies/mlx_inference_distributed

# Clean up
pkill -f "python.*api" || true
pkill -f "python.*inference" || true
sleep 2

# Create simple single-node config
cat > single_node_config.json <<EOF
{
  "model": "mlx-community/Qwen3-1.7B-8bit",
  "device": "mini1",
  "api_port": 8100
}
EOF

# Start the OpenAI-compatible API server in single-node mode
echo "Starting MLX inference server..."
PYTHONUNBUFFERED=1 uv run python -c "
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import mlx.core as mx
from mlx_lm import load, generate
import time

# Load model
print('Loading model...')
model, tokenizer = load('mlx-community/Qwen3-1.7B-8bit')
print('Model loaded!')

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 100
    temperature: float = 0.7

class ChatResponse(BaseModel):
    choices: List[Dict]
    usage: Dict

@app.get('/health')
def health():
    return {'status': 'healthy', 'model': 'mlx-community/Qwen3-1.7B-8bit'}

@app.post('/v1/chat/completions')
async def chat(request: ChatRequest):
    # Build prompt
    prompt = ''
    for msg in request.messages:
        if msg.role == 'system':
            prompt += f'{msg.content}\n\n'
        elif msg.role == 'user':
            prompt += f'User: {msg.content}\nAssistant: '
    
    # Generate
    response = generate(
        model, tokenizer, 
        prompt=prompt,
        max_tokens=request.max_tokens,
        temp=request.temperature
    )
    
    return ChatResponse(
        choices=[{
            'message': {'role': 'assistant', 'content': response},
            'finish_reason': 'stop'
        }],
        usage={'total_tokens': len(tokenizer.encode(prompt + response))}
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8100)
" &

PID=$!
echo "Server PID: $PID"

# Wait for server
echo "Waiting for server to start..."
sleep 15

# Test
echo ""
echo "🧪 Testing server..."
curl -s http://localhost:8100/health | python -m json.tool

echo ""
echo "🎯 READY FOR INFERENCE!"
echo ""
echo "Test with:"
echo 'curl -X POST http://localhost:8100/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 20}'"'"''
echo ""
echo "Stop server: kill $PID"