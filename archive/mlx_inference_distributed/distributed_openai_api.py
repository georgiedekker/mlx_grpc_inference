"""
OpenAI-compatible API server integrated with distributed MLX inference.

This module provides the FastAPI server that connects the existing OpenAI API
endpoints with the new gRPC-based distributed inference backend.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import json
import time
import logging
from typing import List, Dict, Optional, AsyncGenerator, Tuple
import mlx.core as mx

# Import existing OpenAI API models and endpoints
from openai_api import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatMessage, CompletionRequest, CompletionResponse, CompletionChoice,
    Model, ModelList, ChatCompletionStreamResponse, ChatCompletionStreamChoice,
    convert_frequency_penalty_to_repetition
)

# Import distributed inference components
from grpc_client import DistributedInferenceClient, DistributedInferenceOrchestrator
from distributed_config_v2 import DistributedConfig
from device_capabilities import DeviceCapabilityDetector, compare_devices
from sharding_strategy import ResourceAwareShardingPlanner
from model_abstraction import ModelFactory

logger = logging.getLogger(__name__)


class DistributedInferenceAPI:
    """API server that uses distributed inference backend."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.client = DistributedInferenceClient(
            timeout=config.inference.timeout_s,
            max_workers=len(config.devices) * 2
        )
        self.orchestrator = DistributedInferenceOrchestrator(self.client)
        self.model_wrapper = None
        self.tokenizer = None
        self.sharding_plan = None
        self.initialized = False
        
    async def startup(self):
        """Initialize distributed cluster on startup."""
        logger.info("Starting distributed inference API")
        
        try:
            # Load model wrapper to get model info
            self.model_wrapper = ModelFactory.create_wrapper(self.config.model.get_full_name())
            self.model_wrapper.load_model()
            self.tokenizer = self.model_wrapper.tokenizer
            
            # Detect capabilities for local device if not provided
            for device in self.config.devices:
                if not device.capabilities and device.hostname in ["localhost", "127.0.0.1", "mini1.local"]:
                    detector = DeviceCapabilityDetector()
                    device.capabilities = detector.detect_capabilities()
            
            # Create sharding plan
            planner = ResourceAwareShardingPlanner()
            device_profiles = [d.capabilities for d in self.config.get_enabled_devices() 
                             if d.capabilities]
            
            if not device_profiles:
                raise ValueError("No devices with detected capabilities")
            
            self.sharding_plan = planner.create_plan(
                model_info=self.model_wrapper.model_info,
                devices=device_profiles,
                strategy=self.config.sharding.strategy,
                custom_proportions=self.config.sharding.custom_proportions
            )
            
            logger.info("Sharding plan created:")
            self.sharding_plan.print_summary()
            
            # Setup cluster
            device_configs = [
                {
                    "device_id": d.device_id,
                    "hostname": d.hostname,
                    "port": d.port
                }
                for d in self.config.get_enabled_devices()
            ]
            
            self.orchestrator.setup_cluster(
                device_configs=device_configs,
                model_name=self.config.model.get_full_name(),
                model_provider=self.config.model.provider,
                sharding_plan=self.sharding_plan
            )
            
            self.initialized = True
            logger.info("Distributed inference API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed inference: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown distributed cluster."""
        logger.info("Shutting down distributed inference API")
        
        if self.orchestrator:
            self.orchestrator.shutdown()
        
        self.initialized = False
    
    async def generate_distributed(self, prompt: str, 
                                 max_tokens: int = 512,
                                 temperature: float = 0.7,
                                 top_p: float = 0.9,
                                 repetition_penalty: float = 1.1) -> Tuple[str, int]:
        """Generate text using distributed inference.
        
        Returns:
            (generated_text, token_count)
        """
        if not self.initialized:
            raise RuntimeError("Distributed inference not initialized")
        
        # For now, use the orchestrator's simple generate method
        # In production, this would implement proper generation with all parameters
        response = self.orchestrator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Count tokens
        tokens = self.tokenizer.encode(response, add_special_tokens=False)
        
        return response, len(tokens)
    
    async def stream_distributed(self, prompt: str,
                               max_tokens: int = 512,
                               temperature: float = 0.7,
                               top_p: float = 0.9,
                               repetition_penalty: float = 1.1) -> AsyncGenerator[str, None]:
        """Stream text generation using distributed inference."""
        if not self.initialized:
            raise RuntimeError("Distributed inference not initialized")
        
        # Generate full response first (streaming would require pipeline modification)
        response, _ = await self.generate_distributed(
            prompt, max_tokens, temperature, top_p, repetition_penalty
        )
        
        # Simulate streaming by yielding words
        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.01)  # Small delay to simulate streaming


# Global API instance
api_instance: Optional[DistributedInferenceAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global api_instance
    
    # Load configuration
    config_path = "distributed_config.json"
    try:
        config = DistributedConfig.load(config_path)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        from distributed_config_v2 import create_example_config
        config = create_example_config()
    
    # Validate configuration
    is_valid, errors = config.validate()
    if not is_valid:
        logger.error(f"Invalid configuration: {errors}")
        raise ValueError(f"Invalid configuration: {errors}")
    
    # Create API instance
    api_instance = DistributedInferenceAPI(config)
    
    # Startup
    await api_instance.startup()
    
    yield
    
    # Shutdown
    await api_instance.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Distributed MLX OpenAI-Compatible API",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Distributed MLX Inference API",
        "status": "running",
        "mode": "distributed",
        "initialized": str(api_instance.initialized if api_instance else False)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not api_instance or not api_instance.initialized:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Distributed inference not initialized",
                "timestamp": time.time()
            }
        )
    
    # Check device health
    device_status = {}
    for device_id, conn in api_instance.client.connections.items():
        device_status[device_id] = {
            "healthy": conn.healthy,
            "last_check": conn.last_health_check
        }
    
    return {
        "status": "healthy",
        "model": api_instance.config.model.get_full_name(),
        "devices": device_status,
        "sharding_strategy": api_instance.config.sharding.strategy.value,
        "timestamp": time.time()
    }


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models."""
    if not api_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    model_id = api_instance.config.model.get_full_name()
    
    return ModelList(
        data=[
            Model(
                id=model_id,
                created=int(time.time()),
                owned_by=api_instance.config.model.provider,
                root=model_id
            )
        ]
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using distributed inference."""
    if not api_instance or not api_instance.initialized:
        raise HTTPException(status_code=503, detail="Distributed inference not initialized")
    
    try:
        # Convert messages to prompt
        messages = []
        for msg in request.messages:
            if msg.content:
                messages.append({"role": msg.role, "content": msg.content})
        
        if not messages:
            raise ValueError("No valid messages provided")
        
        # Apply chat template
        if api_instance.tokenizer.chat_template:
            prompt = api_instance.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nassistant: "
        
        # Handle streaming
        if request.stream:
            completion_id = f"chatcmpl-{int(time.time())}"
            
            async def stream_response():
                # First chunk with role
                first_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=int(time.time()),
                    model=api_instance.config.model.get_full_name(),
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatMessage(role="assistant"),
                        finish_reason=None
                    )]
                )
                yield f"data: {json.dumps(first_chunk.model_dump())}\n\n"
                
                # Stream content
                async for word in api_instance.stream_distributed(
                    prompt=prompt,
                    max_tokens=request.max_tokens or api_instance.config.inference.default_max_tokens,
                    temperature=request.temperature or api_instance.config.inference.default_temperature,
                    top_p=request.top_p or api_instance.config.inference.default_top_p,
                    repetition_penalty=convert_frequency_penalty_to_repetition(request.frequency_penalty or 0.0)
                ):
                    chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=int(time.time()),
                        model=api_instance.config.model.get_full_name(),
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta=ChatMessage(content=word),
                            finish_reason=None
                        )]
                    )
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                
                # Final chunk
                final_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=int(time.time()),
                    model=api_instance.config.model.get_full_name(),
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatMessage(content=""),
                        finish_reason="stop"
                    )]
                )
                yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        response, token_count = await api_instance.generate_distributed(
            prompt=prompt,
            max_tokens=request.max_tokens or api_instance.config.inference.default_max_tokens,
            temperature=request.temperature or api_instance.config.inference.default_temperature,
            top_p=request.top_p or api_instance.config.inference.default_top_p,
            repetition_penalty=convert_frequency_penalty_to_repetition(request.frequency_penalty or 0.0)
        )
        
        # Count prompt tokens
        prompt_tokens = len(api_instance.tokenizer.encode(prompt, add_special_tokens=True))
        
        return ChatCompletionResponse(
            model=api_instance.config.model.get_full_name(),
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response),
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
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion using distributed inference."""
    if not api_instance or not api_instance.initialized:
        raise HTTPException(status_code=503, detail="Distributed inference not initialized")
    
    try:
        # Handle prompt format
        if isinstance(request.prompt, str):
            prompt = request.prompt
        elif isinstance(request.prompt, list) and all(isinstance(p, str) for p in request.prompt):
            if len(request.prompt) > 1:
                raise ValueError("Multiple prompts not supported in distributed mode")
            prompt = request.prompt[0]
        else:
            raise ValueError("Unsupported prompt format")
        
        # Generate response
        response, token_count = await api_instance.generate_distributed(
            prompt=prompt,
            max_tokens=request.max_tokens or api_instance.config.inference.default_max_tokens,
            temperature=request.temperature or api_instance.config.inference.default_temperature,
            top_p=request.top_p or api_instance.config.inference.default_top_p,
            repetition_penalty=convert_frequency_penalty_to_repetition(request.frequency_penalty or 0.0)
        )
        
        # Count prompt tokens
        prompt_tokens = len(api_instance.tokenizer.encode(prompt, add_special_tokens=True))
        
        # Handle echo option
        text = prompt + response if request.echo else response
        
        return CompletionResponse(
            model=api_instance.config.model.get_full_name(),
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
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/cluster/status")
async def cluster_status():
    """Get detailed cluster status (custom endpoint)."""
    if not api_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Get device statuses
    devices = []
    for device_config in api_instance.config.devices:
        device_info = {
            "device_id": device_config.device_id,
            "hostname": device_config.hostname,
            "port": device_config.port,
            "role": device_config.role,
            "enabled": device_config.enabled
        }
        
        # Add connection status if connected
        if device_config.device_id in api_instance.client.connections:
            conn = api_instance.client.connections[device_config.device_id]
            device_info["connected"] = True
            device_info["healthy"] = conn.healthy
            
            if conn.assignment:
                device_info["shard"] = {
                    "start_layer": conn.assignment.start_layer,
                    "end_layer": conn.assignment.end_layer,
                    "num_layers": conn.assignment.num_layers,
                    "memory_gb": conn.assignment.estimated_memory_gb,
                    "utilization": conn.assignment.memory_utilization()
                }
        else:
            device_info["connected"] = False
        
        if device_config.capabilities:
            device_info["capabilities"] = {
                "model": device_config.capabilities.model,
                "memory_gb": device_config.capabilities.memory_gb,
                "gpu_cores": device_config.capabilities.gpu_cores
            }
        
        devices.append(device_info)
    
    return {
        "cluster_initialized": api_instance.initialized,
        "model": {
            "name": api_instance.config.model.get_full_name(),
            "provider": api_instance.config.model.provider,
            "architecture": api_instance.model_wrapper.model_info.architecture if api_instance.model_wrapper else None,
            "total_layers": api_instance.model_wrapper.model_info.num_layers if api_instance.model_wrapper else None
        },
        "sharding": {
            "strategy": api_instance.config.sharding.strategy.value,
            "balance_score": api_instance.sharding_plan.balance_score if api_instance.sharding_plan else None
        },
        "devices": devices
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )