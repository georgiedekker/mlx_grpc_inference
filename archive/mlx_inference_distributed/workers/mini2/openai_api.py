"""
OpenAI-compatible API implementation for MLX models.

This module provides FastAPI endpoints that match the OpenAI API specification,
allowing the MLX model to be used as a drop-in replacement for OpenAI services.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Literal, AsyncGenerator
from mlx_inference import MLXInference, MLXInferenceError
import json
import time
import uuid
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

app = FastAPI(title="MLX OpenAI-Compatible API", version="1.0.0")

# Global model instance
mlx_inference: Optional[MLXInference] = None
MODEL_ID = "mlx-community/Qwen3-1.7B-8bit"


# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = Field(
        ..., 
        description="The role of the message author",
        example="user"
    )
    content: Optional[str] = Field(
        None,
        description="The content of the message",
        example="Hello! How are you?"
    )
    name: Optional[str] = Field(
        None,
        description="An optional name for the participant",
        example=None
    )
    function_call: Optional[Dict[str, Any]] = Field(
        None,
        description="Function call information",
        example=None
    )


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        ...,
        description="ID of the model to use",
        example="mlx-community/Qwen3-1.7B-8bit"
    )
    messages: List[ChatMessage] = Field(
        ...,
        description="A list of messages comprising the conversation so far",
        example=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi mom!"}
        ]
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0 and 2",
        example=0.7
    )
    top_p: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
        example=1.0
    )
    n: Optional[int] = Field(
        default=1,
        ge=1,
        le=1,
        description="Number of completions to generate",
        example=1
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream back partial progress",
        example=False
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Sequences where the API will stop generating",
        example=None
    )
    max_tokens: Optional[int] = Field(
        default=100,
        ge=1,
        description="Maximum number of tokens to generate",
        example=100
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalize new tokens based on their presence in the text so far",
        example=0.0
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalize new tokens based on their frequency in the text so far",
        example=0.0
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description="Modify the likelihood of specified tokens appearing",
        example=None
    )
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user",
        example=None
    )
    
    @validator('model')
    def validate_model(cls, v):
        # Accept any model name but log if it's different from what we're using
        if v != MODEL_ID:
            logger.info(f"Requested model '{v}' but using '{MODEL_ID}'")
        return v


class CompletionRequest(BaseModel):
    model: str = Field(
        ...,
        description="ID of the model to use",
        example="mlx-community/Qwen3-1.7B-8bit"
    )
    prompt: Union[str, List[str], List[int], List[List[int]]] = Field(
        ...,
        description="The prompt(s) to generate completions for",
        example="Once upon a time"
    )
    suffix: Optional[str] = Field(
        default=None,
        description="Text to append after the completion",
        example=None
    )
    max_tokens: Optional[int] = Field(
        default=100,
        ge=1,
        description="Maximum number of tokens to generate",
        example=100
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
        example=0.7
    )
    top_p: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
        example=1.0
    )
    n: Optional[int] = Field(
        default=1,
        ge=1,
        le=1,
        description="Number of completions to generate",
        example=1
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream back partial progress",
        example=False
    )
    logprobs: Optional[int] = Field(
        default=None,
        description="Include log probabilities",
        example=None
    )
    echo: Optional[bool] = Field(
        default=False,
        description="Echo back the prompt in addition to the completion",
        example=False
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Sequences where the API will stop generating",
        example=None
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalize new tokens based on their presence",
        example=0.0
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalize new tokens based on their frequency",
        example=0.0
    )
    best_of: Optional[int] = Field(
        default=1,
        ge=1,
        description="Generate multiple and return the best",
        example=1
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description="Modify token likelihood",
        example=None
    )
    user: Optional[str] = Field(
        default=None,
        description="User identifier",
        example=None
    )


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "content_filter", "null"]]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "content_filter", "null"]]


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    system_fingerprint: Optional[str] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[Literal["stop", "length"]]


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]
    system_fingerprint: Optional[str] = None


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: str
    parent: Optional[str] = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[Model]


# Helper functions
def convert_frequency_penalty_to_repetition(frequency_penalty: float) -> float:
    """Convert OpenAI's frequency penalty (-2.0 to 2.0) to repetition penalty (1.0+)."""
    # frequency_penalty: -2.0 (less repetition) to 2.0 (more repetition)
    # repetition_penalty: 1.0 (no penalty) to higher (more penalty)
    # We invert the logic: negative frequency penalty means more repetition penalty
    if frequency_penalty <= 0:
        return 1.0 + abs(frequency_penalty) * 0.1  # -2.0 -> 1.2, 0 -> 1.0
    else:
        return max(1.0, 1.0 - frequency_penalty * 0.05)  # 2.0 -> 0.9, but minimum 1.0


async def stream_chat_completion(
    messages: List[Dict[str, str]], 
    request: ChatCompletionRequest,
    completion_id: str
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion responses."""
    try:
        # For now, we'll generate the full response and stream it token by token
        # In a real implementation, you'd want to stream from the model directly
        response, token_count = mlx_inference.chat(
            messages=messages,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=convert_frequency_penalty_to_repetition(request.frequency_penalty),
            return_token_count=True
        )
        
        # Stream the response token by token
        words = response.split()
        
        # First chunk with role
        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"
        
        # Stream content chunks
        for i, word in enumerate(words):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " " if i < len(words) - 1 else word},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)  # Small delay to simulate streaming
        
        # Send final chunk with finish_reason
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "streaming_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


# API endpoints
@app.on_event("startup")
async def startup_event():
    global mlx_inference
    try:
        mlx_inference = MLXInference(MODEL_ID)
        logger.info(f"Model {MODEL_ID} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models."""
    return ModelList(
        data=[
            Model(
                id=MODEL_ID,
                created=int(time.time()),
                owned_by="mlx-community",
                root=MODEL_ID,
                permission=[
                    {
                        "id": "modelperm-" + uuid.uuid4().hex[:12],
                        "object": "model_permission",
                        "created": int(time.time()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ]
            )
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    return Model(
        id=MODEL_ID,
        created=int(time.time()),
        owned_by="mlx-community",
        root=MODEL_ID
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    if mlx_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert messages to the format expected by our inference module
        messages = []
        for msg in request.messages:
            if msg.content:  # Skip messages without content
                messages.append({"role": msg.role, "content": msg.content})
        
        if not messages:
            raise ValueError("No valid messages provided")
        
        # Handle streaming
        if request.stream:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            return StreamingResponse(
                stream_chat_completion(messages, request, completion_id),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        response, token_count = mlx_inference.chat(
            messages=messages,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=convert_frequency_penalty_to_repetition(request.frequency_penalty),
            return_token_count=True
        )
        
        # Count prompt tokens (approximate)
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        prompt_tokens = len(mlx_inference.tokenizer.encode(prompt_text, add_special_tokens=True))
        
        return ChatCompletionResponse(
            model=MODEL_ID,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response),
                    finish_reason="stop" if token_count < (request.max_tokens or 512) else "length"
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MLXInferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    if mlx_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Handle different prompt formats
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        elif isinstance(request.prompt, list) and all(isinstance(p, str) for p in request.prompt):
            prompts = request.prompt
        else:
            raise ValueError("Unsupported prompt format")
        
        # We only support single prompt for now
        if len(prompts) > 1:
            raise ValueError("Multiple prompts not supported")
        
        prompt = prompts[0]
        
        # Generate response
        response, token_count = mlx_inference.generate_response(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=convert_frequency_penalty_to_repetition(request.frequency_penalty),
            return_token_count=True
        )
        
        # Count prompt tokens
        prompt_tokens = len(mlx_inference.tokenizer.encode(prompt, add_special_tokens=True))
        
        # Handle echo option
        text = prompt + response if request.echo else response
        
        return CompletionResponse(
            model=MODEL_ID,
            choices=[
                CompletionChoice(
                    text=text,
                    index=0,
                    finish_reason="stop" if token_count < request.max_tokens else "length"
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MLXInferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if mlx_inference and mlx_inference.is_loaded else "unhealthy",
        "model": MODEL_ID,
        "timestamp": datetime.utcnow().isoformat()
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": "The requested resource was not found",
                "type": "not_found_error",
                "code": "404"
            }
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An internal server error occurred",
                "type": "internal_error",
                "code": "500"
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)