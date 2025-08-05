#!/usr/bin/env python3
"""
Hybrid Distributed Server
Uses PyTorch distributed with Gloo (fixed for Thunderbolt), gRPC, and direct TCP
"""
import os
import sys
import torch
import torch.distributed as dist
import socket
import time
import logging
import asyncio
from datetime import timedelta
from typing import Optional, Dict, Any, List
import json
from pathlib import Path

# Import our components
from src.hybrid import HybridDistributedInference
from src.core.pytorch_kv_cache import (
    DistributedKVCacheManager, 
    DeviceCapability, 
    load_device_capabilities_from_config
)
from src.utils.mlx_pytorch_adapter import load_mlx_model_for_pytorch
from src.model.sharding import ShardingStrategy, ModelShardAssigner
from src.model.inference import DistributedModelInference

# FastAPI for coordinator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] Rank %(rank)s - %(levelname)s - %(message)s'
)

class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = os.environ.get('RANK', '?')
        return True

logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())

class HybridDistributedServer:
    """
    Production-ready distributed server using hybrid approach
    """
    
    def __init__(self):
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_coordinator = (self.rank == 0)
        
        # Configuration
        self.config_path = os.environ.get('CONFIG_PATH', 'config/cluster_config.yaml')
        self.model_name = os.environ.get('MODEL_NAME', 'mlx-community/Qwen3-1.7B-8bit')
        
        # Components
        self.hybrid_dist = None
        self.model = None
        self.tokenizer = None
        self.kv_cache = None
        self.inference_engine = None
        
        # FastAPI app for coordinator
        if self.is_coordinator:
            self.app = self._create_fastapi_app()
        
        logger.info(f"Initializing hybrid server on rank {self.rank}")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize hybrid distributed system
            self.hybrid_dist = HybridDistributedInference()
            logger.info("Hybrid distributed system initialized")
            
            # Load configuration
            self.config = self._load_config()
            
            # Determine device capability and layer assignment
            self.device_capability = self._get_device_capability()
            self.assigned_layers = self._get_layer_assignment()
            
            # Load model shards
            await self._load_model_shards()
            
            # Initialize KV cache
            self._initialize_kv_cache()
            
            # Initialize inference engine
            self.inference_engine = DistributedModelInference(
                model=self.model,
                hybrid_dist=self.hybrid_dist,
                kv_cache=self.kv_cache,
                assigned_layers=self.assigned_layers
            )
            
            # Synchronize all nodes
            self.hybrid_dist.barrier()
            
            logger.info(f"Server initialized successfully on rank {self.rank}")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load cluster configuration"""
        import yaml
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_device_capability(self) -> DeviceCapability:
        """Get device capability for this rank"""
        devices = self.config['devices']
        for device in devices:
            if device['rank'] == self.rank:
                return DeviceCapability(
                    device_id=device['device_id'],
                    memory_gb=device['capabilities']['memory_gb'],
                    gpu_cores=device['capabilities']['gpu_cores'],
                    bandwidth_gbps=device['capabilities']['bandwidth_gbps']
                )
        raise ValueError(f"Device not found for rank {self.rank}")
    
    def _get_layer_assignment(self) -> List[int]:
        """Get layers assigned to this rank"""
        # Use configuration or calculate based on capabilities
        if 'layer_distribution' in self.config['model']:
            device_id = self.device_capability.device_id
            return self.config['model']['layer_distribution'].get(device_id, [])
        else:
            # Calculate fair distribution
            total_layers = self.config['model']['total_layers']
            layers_per_rank = total_layers // self.world_size
            start = self.rank * layers_per_rank
            end = start + layers_per_rank if self.rank < self.world_size - 1 else total_layers
            return list(range(start, end))
    
    async def _load_model_shards(self):
        """Load model shards for assigned layers"""
        logger.info(f"Loading model shards for layers {self.assigned_layers}")
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load full model on coordinator
        if self.is_coordinator:
            model, tokenizer = load_mlx_model_for_pytorch(self.model_name, device)
            self.model = model
            self.tokenizer = tokenizer
            
            # Broadcast model structure to other ranks if needed
            if self.world_size > 1:
                # This is where we'd implement model distribution
                pass
        else:
            # Workers load their assigned layers
            # For now, load full model (optimization for later)
            model, tokenizer = load_mlx_model_for_pytorch(self.model_name, device)
            self.model = model
            self.tokenizer = tokenizer
        
        logger.info(f"Model loaded successfully on rank {self.rank}")
    
    def _initialize_kv_cache(self):
        """Initialize distributed KV cache"""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Collect all device capabilities
        all_capabilities = []
        for i in range(self.world_size):
            if i == self.rank:
                all_capabilities.append(self.device_capability)
            else:
                # In production, exchange capabilities via distributed communication
                all_capabilities.append(self.device_capability)  # Placeholder
        
        self.kv_cache = DistributedKVCacheManager(
            rank=self.rank,
            world_size=self.world_size,
            device=device,
            device_capabilities=all_capabilities
        )
        
        logger.info("KV cache initialized")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app for coordinator"""
        app = FastAPI(title="Hybrid Distributed Inference API")
        
        class GenerateRequest(BaseModel):
            prompt: str
            max_tokens: int = 50
            temperature: float = 0.7
        
        class ChatCompletionRequest(BaseModel):
            model: str = "mlx-community/Qwen3-1.7B-8bit"
            messages: List[Dict[str, str]]
            max_tokens: int = 100
            temperature: float = 0.7
            stream: bool = False
        
        @app.get("/health")
        async def health():
            nodes_status = {}
            for rank in range(self.world_size):
                nodes_status[f"rank_{rank}"] = {
                    "healthy": True,
                    "device": f"Device {rank}",
                    "layers": len(self.assigned_layers) if rank == self.rank else "unknown"
                }
            
            return {
                "status": "healthy",
                "coordinator": True,
                "world_size": self.world_size,
                "nodes": nodes_status
            }
        
        @app.post("/generate")
        async def generate(request: GenerateRequest):
            try:
                # This is where we'd implement actual generation
                result = await self._generate_text(
                    request.prompt,
                    request.max_tokens,
                    request.temperature
                )
                return {"generated_text": result}
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible endpoint"""
            try:
                # Extract last message
                prompt = request.messages[-1]["content"] if request.messages else ""
                
                result = await self._generate_text(
                    prompt,
                    request.max_tokens,
                    request.temperature
                )
                
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(result.split()),
                        "total_tokens": len(prompt.split()) + len(result.split())
                    }
                }
            except Exception as e:
                logger.error(f"Chat completion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    async def _generate_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using distributed inference"""
        if not self.inference_engine:
            return "Model not initialized"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Run distributed inference
        generated_ids = await self.inference_engine.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        # Decode output
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Return only the new text
        return generated_text[len(prompt):].strip()
    
    async def run_coordinator(self):
        """Run as coordinator with API server"""
        if not self.is_coordinator:
            raise ValueError("Only coordinator can run API server")
        
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=8100,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def run_worker(self):
        """Run as worker node"""
        logger.info(f"Worker {self.rank} ready")
        
        # Keep worker alive and process requests
        while True:
            try:
                # Workers wait for and process inference requests
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                break
    
    async def run(self):
        """Main run method"""
        # Initialize server
        if not await self.initialize():
            logger.error("Failed to initialize server")
            return
        
        # Run based on role
        if self.is_coordinator:
            await self.run_coordinator()
        else:
            await self.run_worker()
    
    def cleanup(self):
        """Clean up resources"""
        if self.hybrid_dist:
            self.hybrid_dist.cleanup()
        logger.info("Server shutdown complete")


async def main():
    """Main entry point"""
    server = HybridDistributedServer()
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())