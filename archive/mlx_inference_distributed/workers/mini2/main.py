from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from mlx_inference import MLXInference
import uvicorn


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class ChatResponse(BaseModel):
    response: str
    model: str = "mlx-community/Qwen3-1.7B-8bit"
    tokens_used: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


app = FastAPI(title="MLX Inference API", version="1.0.0")

mlx_inference: Optional[MLXInference] = None


@app.on_event("startup")
async def startup_event():
    global mlx_inference
    try:
        mlx_inference = MLXInference()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "MLX Inference API", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if mlx_inference is not None else "unhealthy",
        model_loaded=mlx_inference is not None,
        model_name="mlx-community/Qwen3-1.7B-8bit"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if mlx_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        response, token_count = mlx_inference.chat(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            return_token_count=True
        )
        
        return ChatResponse(response=response, tokens_used=token_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/generate", response_model=ChatResponse)
async def generate(request: GenerateRequest):
    if mlx_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        response, token_count = mlx_inference.generate_response(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            return_token_count=True
        )
        
        return ChatResponse(response=response, tokens_used=token_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)