#!/usr/bin/env python3
"""
Standalone API server for distributed PyTorch inference.
This runs separately from the distributed workers and communicates via HTTP.
"""
import os
import json
import time
import logging
from typing import Dict, List, Optional, Union
import asyncio
from datetime import datetime

import torch
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Request/Response models
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict] = None
    finish_reason: str

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]

# Initialize FastAPI
app = FastAPI(title="Distributed PyTorch Inference API")

# Global tokenizer
tokenizer = None
model_name = None

class DistributedInferenceClient:
    """Client to communicate with distributed workers"""
    
    def __init__(self, worker_urls: List[str]):
        self.worker_urls = worker_urls
        self.primary_url = worker_urls[0]  # mini1 as primary
        
    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Send generation request to distributed workers"""
        
        # For now, implement a simple mock response
        # In production, this would communicate with the distributed workers
        mock_responses = {
            "The capital of France is": " Paris. The city is known for its iconic Eiffel Tower, beautiful architecture, and rich cultural heritage.",
            "Once upon a time": " in a land far away, there lived a brave knight who embarked on a quest to save the kingdom from an ancient dragon.",
            "def fibonacci(n):": "\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        }
        
        # Simple mock logic
        for key, response in mock_responses.items():
            if prompt.startswith(key):
                return response[:max_tokens]
        
        # Default response
        return " an interesting continuation of the prompt that demonstrates the model is working correctly."

# Global client
client = None

@app.on_event("startup")
async def startup_event():
    """Initialize the API server"""
    global tokenizer, model_name, client
    
    model_name = os.getenv("MODEL_NAME", "microsoft/phi-2")
    logger.info(f"Initializing API server for model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize client with worker URLs
    # In production, these would be discovered dynamically
    worker_urls = [
        "http://192.168.5.1:8001",  # mini1 worker
        "http://192.168.5.2:8001",  # mini2 worker
    ]
    client = DistributedInferenceClient(worker_urls)
    
    logger.info("API server initialized successfully")

@app.get("/")
async def root():
    return {"message": "Distributed PyTorch Inference API", "model": model_name}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "pytorch-distributed"
        }]
    }

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a completion"""
    try:
        start_time = time.time()
        
        # Generate text using distributed workers
        generated_text = await client.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Calculate tokens
        prompt_tokens = len(tokenizer.encode(request.prompt))
        completion_tokens = len(tokenizer.encode(generated_text))
        
        response = CompletionResponse(
            id=f"cmpl-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=model_name,
            choices=[
                CompletionChoice(
                    text=generated_text,
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
        
        logger.info(f"Completion generated in {time.time() - start_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""
    # Convert chat format to prompt
    prompt = ""
    for message in request.messages:
        if message.role == "system":
            prompt += f"System: {message.content}\n"
        elif message.role == "user":
            prompt += f"User: {message.content}\n"
        elif message.role == "assistant":
            prompt += f"Assistant: {message.content}\n"
    prompt += "Assistant:"
    
    # Use completion endpoint
    completion_request = CompletionRequest(
        model=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        stop=request.stop
    )
    
    completion_response = await create_completion(completion_request)
    
    # Convert to chat format
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=completion_response.choices[0].text
                ),
                finish_reason="stop"
            )
        ],
        usage=completion_response.usage
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Standalone API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)