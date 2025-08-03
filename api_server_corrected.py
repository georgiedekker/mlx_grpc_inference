#!/usr/bin/env python3
"""
Corrected API server that uses proper prompt formatting and generation.
"""

import asyncio
import time
import uuid
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_lm import load, generate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None

# API Models (OpenAI-compatible)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mlx-community/Qwen3-1.7B-8bit"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int  
    total_tokens: int
    prompt_tokens_per_second: Optional[float] = None
    generation_tokens_per_second: Optional[float] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global model, tokenizer
    
    logger.info("🚀 Starting Corrected MLX Inference API...")
    
    # Load model and tokenizer
    logger.info("📦 Loading model: mlx-community/Qwen3-1.7B-8bit")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    logger.info(f"✅ Model loaded: {len(model.layers)} layers")
    
    logger.info("✅ Corrected API server ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down API server...")

app = FastAPI(
    lifespan=lifespan, 
    title="MLX Corrected Inference API",
    description="OpenAI-compatible API with proper prompt formatting", 
    version="1.1.0"
)

def format_messages_for_generation(messages: List[ChatMessage]) -> str:
    """Format messages properly for the model."""
    # For single user message, use direct format
    if len(messages) == 1 and messages[0].role == "user":
        return messages[0].content
    
    # For chat format, use Human/Assistant
    formatted = ""
    for msg in messages:
        if msg.role == "user":
            formatted += f"Human: {msg.content}\\n"
        elif msg.role == "assistant":
            formatted += f"Assistant: {msg.content}\\n"
    
    # Add Assistant: prompt for continuation
    if not formatted.endswith("Assistant: "):
        formatted += "Assistant:"
    
    return formatted

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {"message": "MLX Corrected Inference API", "version": "1.1.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": "corrected_prompt_format",
        "model_loaded": model is not None
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion with corrected prompt formatting."""
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        overall_start_time = time.time()
        
        # Format prompt correctly
        prompt = format_messages_for_generation(request.messages)
        prompt_tokens = len(tokenizer.encode(prompt))
        
        logger.info(f"🚀 Generating response for {prompt_tokens} prompt tokens")
        logger.info(f"📝 Formatted prompt: {repr(prompt[:100])}...")
        
        # Since generate doesn't take temperature, we'll use the manual approach
        generation_start = time.time()
        
        # For single message without role format, use generate directly
        if len(request.messages) == 1 and request.messages[0].role == "user" and not any(role in prompt for role in ["Human:", "Assistant:", "user:", "assistant:"]):
            # Use mlx_lm generate for simple prompts
            response_text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=min(request.max_tokens or 50, 100)
            )
        else:
            # For chat format, we need manual generation with temperature control
            import mlx.core as mx
            from mlx_lm.sample_utils import make_sampler
            
            # Tokenize and process
            input_ids = mx.array(tokenizer.encode(prompt))
            if len(input_ids.shape) == 1:
                input_ids = input_ids[None, :]
            
            # Forward pass through model
            hidden_states = model.model.embed_tokens(input_ids)
            for layer in model.model.layers:
                layer_output = layer(hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
            
            hidden_states = model.model.norm(hidden_states)
            logits = model.model.embed_tokens.as_linear(hidden_states)
            
            # Generate tokens with temperature
            sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
            generated_ids = []
            current_ids = input_ids
            
            for _ in range(min(request.max_tokens or 50, 100)):
                next_token = sampler(logits[:, -1:, :])
                token_id = next_token.item()
                generated_ids.append(token_id)
                
                if token_id == tokenizer.eos_token_id:
                    break
                    
                # Update for next iteration
                new_token_tensor = mx.array([[token_id]])
                current_ids = mx.concatenate([current_ids, new_token_tensor], axis=1)
                
                # Forward pass for new sequence
                hidden_states = model.model.embed_tokens(current_ids)
                for layer in model.model.layers:
                    layer_output = layer(hidden_states)
                    if isinstance(layer_output, tuple):
                        hidden_states = layer_output[0]
                    else:
                        hidden_states = layer_output
                
                hidden_states = model.model.norm(hidden_states)
                logits = model.model.embed_tokens.as_linear(hidden_states)
            
            response_text = tokenizer.decode(generated_ids)
        
        generation_time = time.time() - generation_start
        
        # Calculate tokens
        if isinstance(response_text, str) and response_text.startswith(prompt):
            # If generate returned the full text including prompt
            response_only = response_text[len(prompt):]
            completion_tokens = len(tokenizer.encode(response_only))
        else:
            # If we got just the response
            completion_tokens = len(tokenizer.encode(response_text)) if response_text else 1
        
        # Calculate performance metrics
        total_time = time.time() - overall_start_time
        generation_tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
        prompt_tokens_per_second = prompt_tokens / (total_time - generation_time) if (total_time - generation_time) > 0 else 0
        
        logger.info(f"✅ Generated response:")
        logger.info(f"   Prompt: {prompt_tokens} tokens @ {prompt_tokens_per_second:.1f} tok/s")
        logger.info(f"   Response: {completion_tokens} tokens @ {generation_tokens_per_second:.1f} tok/s")
        logger.info(f"   Content: {repr(response_text[:100])}...")
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_per_second=round(prompt_tokens_per_second, 1),
                generation_tokens_per_second=round(generation_tokens_per_second, 1)
            )
        )
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Error in corrected completion: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "mlx-community/Qwen3-1.7B-8bit",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-community",
                "mode": "corrected_format"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)