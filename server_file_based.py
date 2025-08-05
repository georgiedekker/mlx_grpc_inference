#!/usr/bin/env python3
"""
File-Based Distributed Server for Thunderbolt Bridge Networks
Bypasses PyTorch distributed's Gloo backend issues
"""
import os
import sys
import torch
import socket
import time
import logging
import asyncio
from datetime import timedelta
from typing import Optional, Dict, Any, List
import json
from pathlib import Path

# Import our coordination system
from src.coordination.file_based_coordinator import DistributedInferenceManager

# Import other components
from src.core.pytorch_kv_cache import (
    DistributedKVCacheManager, 
    DeviceCapability, 
    HeterogeneousCacheAllocator,
    load_device_capabilities_from_config,
    estimate_kv_cache_memory
)
from src.utils.mlx_pytorch_adapter import load_mlx_model_for_pytorch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] Rank %(rank)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = os.environ.get('RANK', '?')
        return True

logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())

class FileBasedDistributedServer:
    """
    Distributed inference server using file-based coordination
    Works around Gloo backend limitations on Thunderbolt Bridge
    """
    
    def __init__(self, rank: int, world_size: int, config_path: str):
        self.rank = rank
        self.world_size = world_size
        self.config_path = config_path
        self.is_coordinator = (rank == 0)
        
        # Initialize file-based coordination
        self.manager = DistributedInferenceManager(rank, world_size, config_path)
        
        # Model and caching components
        self.model = None
        self.kv_cache = None
        self.tokenizer = None
        self.device_capability = None
        
        # FastAPI server (coordinator only)
        self.app = None
        
        logger.info(f"Initialized file-based server: rank={rank}, world_size={world_size}")
    
    async def initialize(self) -> bool:
        """Initialize the distributed server"""
        try:
            # Initialize coordination system
            logger.info("Initializing file-based coordination...")
            if not self.manager.initialize():
                logger.error("Failed to initialize coordination system")
                return False
            
            # Load device capabilities
            logger.info("Loading device capabilities...")
            self.device_capability = self._load_device_capability()
            
            # Load model layers assigned to this rank
            logger.info("Loading assigned model layers...")
            await self._load_model_layers()
            
            # Initialize KV cache
            logger.info("Initializing KV cache...")
            self._initialize_kv_cache()
            
            # Setup API server (coordinator only)
            if self.is_coordinator:
                logger.info("Setting up FastAPI server...")
                self._setup_api_server()
            
            logger.info("Server initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            return False
    
    def _load_device_capability(self) -> DeviceCapability:
        """Load device capability for this rank"""
        import yaml
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Find our device in the config
        for device in config['devices']:
            if device['rank'] == self.rank:
                return DeviceCapability(
                    device_id=device['device_id'],
                    memory_gb=device['capabilities']['memory_gb'],
                    gpu_cores=device['capabilities']['gpu_cores'],
                    bandwidth_gbps=device['capabilities']['bandwidth_gbps']
                )
        
        raise ValueError(f"Device configuration not found for rank {self.rank}")
    
    async def _load_model_layers(self):
        """Load only the model layers assigned to this rank"""
        assigned_layers = self.manager.coordinator.get_assigned_layers()
        
        if not assigned_layers:
            logger.error("No layers assigned to this rank")
            return
        
        logger.info(f"Loading layers {assigned_layers}")
        
        # Load the MLX model and convert to PyTorch
        # This integrates with your existing MLX-PyTorch adapter
        import yaml
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_name = config['model']['name']
        
        try:
            # Use your existing adapter to load the model
            logger.info(f"Loading MLX model: {model_name}")
            
            # Determine device (MPS for Mac)
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Load model with device parameter
            model, tokenizer = load_mlx_model_for_pytorch(model_name, device)
            
            # Store the model and tokenizer
            self.model = model  # In future, extract only assigned layers
            self.tokenizer = tokenizer
            
            logger.info(f"Successfully loaded model with {len(assigned_layers)} assigned layers")
            
        except Exception as e:
            logger.error(f"Failed to load model layers: {e}")
            raise
    
    
    def _initialize_kv_cache(self):
        """Initialize KV cache for this rank"""
        try:
            # Create cache allocator with list of capabilities
            allocator = HeterogeneousCacheAllocator([self.device_capability])
            
            # Get model config for Qwen3-1.7B
            model_config = {
                "hidden_size": 1536,  # Qwen3-1.7B hidden size
                "num_attention_heads": 12,
                "num_key_value_heads": 12  # GQA heads
            }
            
            # Calculate cache allocation
            allocations = allocator.calculate_cache_allocation(
                max_batch_size=1,
                sequence_length=2048
            )
            
            # Get allocation for our device
            device_allocation = allocations.get(self.device_capability.device_id, {})
            cache_size = device_allocation.get('max_cached_sequences', 10)
            
            # Initialize cache manager
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.kv_cache = DistributedKVCacheManager(
                rank=self.rank,
                world_size=self.world_size,
                device=device,
                device_capabilities=[self.device_capability]
            )
            
            assigned_layers = self.manager.coordinator.get_assigned_layers()
            logger.info(f"Initialized KV cache with size {cache_size} for {len(assigned_layers)} layers")
            
        except Exception as e:
            logger.error(f"Failed to initialize KV cache: {e}")
            raise
    
    def _setup_api_server(self):
        """Setup FastAPI server (coordinator only)"""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        
        self.app = FastAPI(title="MLX Distributed Inference API")
        
        class GenerateRequest(BaseModel):
            prompt: str
            max_tokens: int = 50
            temperature: float = 0.7
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "rank": self.rank,
                "world_size": self.world_size,
                "coordination": "file-based",
                "nodes": self.manager.coordinator.get_all_nodes()
            }
        
        @self.app.post("/generate")
        async def generate_text(request: GenerateRequest):
            try:
                # This would implement distributed generation
                # across all ranks using your file-based coordination
                result = await self._distributed_generate(
                    request.prompt, 
                    request.max_tokens, 
                    request.temperature
                )
                return {"generated_text": result}
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/cache/stats")
        async def cache_stats():
            if self.kv_cache:
                return {
                    "status": "KV cache initialized",
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "device": str(self.kv_cache.device)
                }
            return {"error": "KV cache not initialized"}
        
        @self.app.post("/cache/clear")
        async def clear_cache():
            if self.kv_cache:
                if hasattr(self.kv_cache, 'clear_cache'):
                    self.kv_cache.clear_cache()
                return {"status": "cache cleared"}
            return {"error": "KV cache not initialized"}
        
        # OpenAI-compatible endpoints
        class ChatCompletionRequest(BaseModel):
            model: str = "mlx-community/Qwen3-1.7B-8bit"
            messages: List[Dict[str, str]]
            temperature: float = 0.7
            max_tokens: int = 100
            stream: bool = False
        
        class ChatCompletionResponse(BaseModel):
            id: str = "chatcmpl-" + str(int(time.time()))
            object: str = "chat.completion"
            created: int = int(time.time())
            model: str
            choices: List[Dict[str, Any]]
            usage: Dict[str, int]
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint"""
            try:
                # Extract the last user message as prompt
                prompt = ""
                for msg in request.messages:
                    if msg["role"] == "user":
                        prompt = msg["content"]
                
                # Generate response
                result = await self._distributed_generate(
                    prompt, 
                    request.max_tokens, 
                    request.temperature
                )
                
                # Format as OpenAI response
                response = ChatCompletionResponse(
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result
                        },
                        "finish_reason": "stop"
                    }],
                    usage={
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(result.split()),
                        "total_tokens": len(prompt.split()) + len(result.split())
                    }
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Chat completion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models"""
            return {
                "object": "list",
                "data": [{
                    "id": "mlx-community/Qwen3-1.7B-8bit",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mlx-community",
                    "permission": [],
                    "root": "mlx-community/Qwen3-1.7B-8bit",
                    "parent": None
                }]
            }
    
    async def _distributed_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Perform distributed text generation across all ranks"""
        # This is where you'd implement the actual distributed inference
        # using your file-based coordination system
        
        logger.info(f"Starting distributed generation: '{prompt[:50]}...'")
        
        # Placeholder implementation
        # In practice, this would:
        # 1. Tokenize the prompt
        # 2. Coordinate with other ranks via file system
        # 3. Process tokens through the distributed model layers
        # 4. Return generated text
        
        return f"Generated response to: {prompt} [File-based distributed inference]"
    
    async def run_coordinator(self, host: str = "0.0.0.0", port: int = 8100):
        """Run the coordinator with API server"""
        if not self.is_coordinator:
            raise ValueError("Only coordinator can run API server")
        
        if not self.app:
            raise ValueError("API server not initialized")
        
        import uvicorn
        logger.info(f"Starting API server on {host}:{port}")
        
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def run_worker(self):
        """Run as worker node"""
        logger.info("Running as worker node")
        
        # Workers just wait and respond to coordination requests
        while True:
            try:
                # Check coordination status
                nodes = self.manager.coordinator.get_all_nodes()
                coordinator_node = nodes.get(0)
                
                if coordinator_node and coordinator_node.status == "shutdown":
                    logger.info("Coordinator shutdown detected, exiting")
                    break
                
                # Update our status
                self.manager.coordinator.update_status("ready")
                
                # Wait a bit before next check
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def run(self):
        """Main run method"""
        try:
            # Initialize the server
            if not await self.initialize():
                logger.error("Server initialization failed")
                return False
            
            # Run based on role
            if self.is_coordinator:
                await self.run_coordinator()
            else:
                await self.run_worker()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the server"""
        logger.info("Shutting down server...")
        if self.manager:
            self.manager.shutdown()

async def main():
    """Main entry point"""
    # Get environment variables with proper error handling
    try:
        rank_str = os.environ.get('RANK', '0').strip()
        world_size_str = os.environ.get('WORLD_SIZE', '1').strip()
        
        # Clean any ANSI escape sequences
        import re
        rank_str = re.sub(r'\x1b\[[0-9;]*m', '', rank_str)
        world_size_str = re.sub(r'\x1b\[[0-9;]*m', '', world_size_str)
        
        rank = int(rank_str)
        world_size = int(world_size_str)
        config_path = os.environ.get('CONFIG_PATH', 'config/cluster_config.yaml')
        
        logger.info(f"Parsed environment: RANK={rank}, WORLD_SIZE={world_size}")
        
    except ValueError as e:
        logger.error(f"Failed to parse environment variables: {e}")
        logger.error(f"RANK='{os.environ.get('RANK', 'NOT_SET')}'")
        logger.error(f"WORLD_SIZE='{os.environ.get('WORLD_SIZE', 'NOT_SET')}'")
        return
    
    # Create and run server
    server = FileBasedDistributedServer(rank, world_size, config_path)
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())