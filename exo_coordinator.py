#!/usr/bin/env python3
"""
Coordinator for distributed inference using exo's approach
Provides OpenAI-compatible API and orchestrates the pipeline
"""
import asyncio
import time
import logging
import uuid
from typing import Dict, Any, List
import numpy as np
import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7


class Coordinator:
    """Coordinates distributed inference across nodes"""
    
    def __init__(self, first_node_address: str):
        self.first_node_address = first_node_address
        # We'll get tokenizer from the first node
        
    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Generate text using the distributed pipeline"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # For now, we'll use a simple tokenizer locally
        # In production, you'd get this from the first node
        from mlx_lm.tokenizer_utils import load_tokenizer
        from pathlib import Path
        
        # Get tokenizer from cached model
        model_name = "mlx-community/Qwen3-1.7B-8bit"
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_id = f"models--{model_name.replace('/', '--')}"
        model_path = cache_dir / model_id / "snapshots"
        if model_path.exists():
            model_path = sorted([d for d in model_path.iterdir() if d.is_dir()])[-1]
            tokenizer = load_tokenizer(model_path)
        else:
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt)
        prompt_array = np.array(prompt_tokens).reshape(1, -1)
        
        logger.info(f"Generating up to {max_tokens} tokens from {len(prompt_tokens)} prompt tokens")
        
        generated_tokens = []
        current_input = prompt_array
        
        # Generate tokens one by one
        for i in range(max_tokens):
            # Send through pipeline
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "tensor_data": base64.b64encode(current_input.tobytes()).decode(),
                    "shape": list(current_input.shape),
                    "dtype": str(current_input.dtype),
                    "request_id": request_id
                }
                
                async with session.post(
                    f"http://{self.first_node_address}/process",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"Pipeline error: {error}")
                        break
                    
                    result = await resp.json()
                    
                    # Decode output tensor (logits)
                    logits_bytes = base64.b64decode(result["tensor_data"])
                    logits = np.frombuffer(logits_bytes, dtype=result["dtype"]).reshape(result["shape"])
                    
                    # Sample next token (simple argmax for now)
                    # In production, you'd use proper sampling with temperature
                    if temperature == 0:
                        next_token = int(np.argmax(logits[0, -1, :]))
                    else:
                        # Simple temperature sampling
                        import mlx.core as mx
                        from mlx_lm.sample_utils import make_sampler
                        sampler = make_sampler(temp=temperature)
                        logits_mx = mx.array(logits)
                        token = sampler(logits_mx[:, -1, :])
                        mx.eval(token)
                        next_token = int(token.item())
                    
                    generated_tokens.append(next_token)
                    
                    # Check for EOS
                    if next_token == tokenizer.eos_token_id:
                        break
                    
                    # Prepare next input (just the new token)
                    current_input = np.array([[next_token]])
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens)
        
        # Calculate metrics
        total_time = time.time() - start_time
        tokens_per_second = len(generated_tokens) / total_time if total_time > 0 else 0
        
        logger.info(f"✅ Generated {len(generated_tokens)} tokens in {total_time:.2f}s")
        logger.info(f"✅ Speed: {tokens_per_second:.1f} tokens/sec")
        
        return {
            "text": generated_text,
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": len(generated_tokens),
            "total_tokens": len(prompt_tokens) + len(generated_tokens),
            "time_taken": total_time,
            "tokens_per_second": tokens_per_second
        }


# Create FastAPI app
app = FastAPI(title="MLX Distributed Inference (Exo-style)")
coordinator = None


@app.on_event("startup")
async def startup():
    global coordinator
    # First node is on mini1
    coordinator = Coordinator("192.168.5.1:50051")
    logger.info("Coordinator initialized")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    try:
        # Format prompt from messages
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        prompt += "Assistant: "
        
        # Generate response
        result = await coordinator.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["generated_tokens"],
                "total_tokens": result["total_tokens"]
            }
        }
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Check health of all nodes"""
    nodes_status = []
    
    # Check mini1 node
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://192.168.5.1:50051/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    nodes_status.append(await resp.json())
    except:
        nodes_status.append({"node_id": "mini1", "status": "offline"})
    
    # Check mini2 node
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://192.168.5.2:50051/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    nodes_status.append(await resp.json())
    except:
        nodes_status.append({"node_id": "mini2", "status": "offline"})
    
    return {
        "status": "healthy" if len(nodes_status) == 2 else "degraded",
        "nodes": nodes_status
    }


@app.get("/")
async def root():
    """Root endpoint with dashboard"""
    html = """
    <html>
        <head>
            <title>MLX Distributed Inference (Exo-style)</title>
            <style>
                body { font-family: Arial; margin: 40px; background: #f0f0f0; }
                .container { background: white; padding: 20px; border-radius: 10px; }
                h1 { color: #333; }
                .status { background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .node { background: #f8f8f8; padding: 10px; margin: 10px 0; border-left: 4px solid #4CAF50; }
                code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 MLX Distributed Inference</h1>
                <p>Using Exo's proven architecture</p>
                
                <div class="status">
                    <h2>Architecture</h2>
                    <div class="node">
                        <strong>mini1 (192.168.5.1:50051)</strong><br>
                        Layers 0-13 + embeddings<br>
                        ~0.85 GB memory
                    </div>
                    <div class="node">
                        <strong>mini2 (192.168.5.2:50051)</strong><br>
                        Layers 14-27 + LM head<br>
                        ~0.85 GB memory
                    </div>
                </div>
                
                <h2>API Endpoint</h2>
                <code>POST http://localhost:8100/v1/chat/completions</code>
                
                <h2>Test Command</h2>
                <pre>
curl -X POST http://localhost:8100/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'
                </pre>
            </div>
        </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")