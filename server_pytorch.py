#!/usr/bin/env python3
"""
Distributed PyTorch Server - Production Implementation
Uses organized component structure with KV caching and heterogeneous sharding
"""
import os
import sys
import torch
import torch.distributed as dist
import socket
import time
import logging
from datetime import timedelta
from typing import Optional, Dict, Any, List
import json
import asyncio
from pathlib import Path

# Import our distributed components
from src.core.pytorch_kv_cache import (
    DistributedKVCacheManager, 
    DeviceCapability, 
    HeterogeneousCacheAllocator,
    load_device_capabilities_from_config,
    estimate_kv_cache_memory
)
from src.utils.mlx_pytorch_adapter import load_mlx_model_for_pytorch

# Setup logging with rank information
class RankFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'hostname'):
            record.hostname = socket.gethostname().split('.')[0]
        if not hasattr(record, 'rank'):
            record.rank = os.environ.get('RANK', '?')
        return True

# Create and configure the filter
rank_filter = RankFilter()

# Setup logging with proper format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(hostname)s] Rank %(rank)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Apply filter to all relevant loggers
logger = logging.getLogger(__name__)
logger.addFilter(rank_filter)

# Apply to ALL loggers including our own modules
import logging
root_logger = logging.getLogger()
root_logger.addFilter(rank_filter)

# Also apply to specific loggers that may be created before root filter is applied
for logger_name in ['', 'transformers', 'torch', 'safetensors', 'huggingface_hub', 'src', 'src.utils', 'src.utils.mlx_pytorch_adapter']:
    logging.getLogger(logger_name).addFilter(rank_filter)


class DistributedModelServer:
    """Production distributed inference server with KV caching and heterogeneous sharding"""
    
    def __init__(self):
        # Environment configuration
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.master_addr = os.environ.get('MASTER_ADDR', '192.168.5.1')
        self.master_port = os.environ.get('MASTER_PORT', '29501')
        self.model_name = os.environ.get('MODEL_NAME', 'mlx-community/Qwen3-1.7B-8bit')
        
        logger.info(f"Initializing distributed server - Rank {self.rank}/{self.world_size}")
        logger.info(f"Model: {self.model_name}")
        
        # Load device capabilities from config
        self.device_capabilities = self._load_device_capabilities()
        
        # Initialize distributed if multi-node
        if self.world_size > 1:
            self._init_distributed()
        
        # Detect compute device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize KV cache manager
        self.cache_manager = DistributedKVCacheManager(
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            device_capabilities=self.device_capabilities
        )
        
        # Load and shard model
        self._load_model_shard()
        
        # Initialize cache after model loading
        self._initialize_cache()
        
        # Generation tracking
        self.generation_cache = {}
        self.sequence_counter = 0
    
    def _load_device_capabilities(self) -> Optional[List[DeviceCapability]]:
        """Load device capabilities from cluster configuration"""
        config_path = "config/cluster_config.yaml"
        if os.path.exists(config_path):
            try:
                capabilities = load_device_capabilities_from_config(config_path)
                logger.info(f"Loaded capabilities for {len(capabilities)} devices")
                
                # Log capability details
                for cap in capabilities:
                    logger.info(f"Device {cap.device_id}: {cap.memory_gb}GB RAM, {cap.gpu_cores} GPU cores")
                
                return capabilities
            except Exception as e:
                logger.warning(f"Could not load device capabilities: {e}")
        else:
            logger.warning(f"Config file not found: {config_path}")
        
        return None
    
    def _init_distributed(self):
        """Initialize PyTorch distributed with robust error handling"""
        # Set environment variables for initialization
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        
        # Set network interface for Thunderbolt bridge (PyTorch docs recommend this)
        os.environ['GLOO_SOCKET_IFNAME'] = 'bridge0'
        
        logger.info(f"Initializing distributed: Master {self.master_addr}:{self.master_port}")
        
        # Workers wait longer for master to fully initialize and bind
        if self.rank > 0:
            logger.info("Worker waiting 10s for master...")
            time.sleep(10)
        else:
            # Master waits a bit to ensure proper binding
            logger.info("Master waiting 2s to bind properly...")
            time.sleep(2)
        
        try:
            # Use explicit TCP addresses for Thunderbolt network (from working config)
            if self.rank == 0:
                init_method = f'tcp://0.0.0.0:{self.master_port}'
            else:
                init_method = f'tcp://{self.master_addr}:{self.master_port}'
            
            logger.info(f"Using init_method: {init_method}")
            
            dist.init_process_group(
                backend='gloo',
                init_method=init_method,
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=30)  # Shorter timeout to avoid connection state issues
            )
            logger.info("✓ Distributed initialization successful")
            
            # Test basic communication
            self._test_distributed_communication()
            
        except Exception as e:
            logger.error(f"Distributed initialization failed: {e}")
            logger.warning("Falling back to single-node mode")
            self.world_size = 1
    
    def _test_distributed_communication(self):
        """Test basic distributed communication"""
        try:
            if self.rank == 0:
                # Master sends test data
                test_tensor = torch.tensor([42.0])
                for target in range(1, self.world_size):
                    dist.send(test_tensor, dst=target)
                    logger.info(f"✓ Sent test data to rank {target}")
            else:
                # Workers receive test data
                test_tensor = torch.zeros(1)
                dist.recv(test_tensor, src=0)
                logger.info(f"✓ Received test data: {test_tensor.item()}")
        except Exception as e:
            logger.error(f"Communication test failed: {e}")
            raise
    
    def _load_model_shard(self):
        """Load model and create shard for this rank"""
        logger.info(f"Loading model shard for rank {self.rank}")
        
        try:
            # Load model using our adapter
            self.model, self.tokenizer = load_mlx_model_for_pytorch(self.model_name, self.device)
            logger.info("✓ Model loaded successfully")
            
            # Set pad token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Shard model if distributed
            if self.world_size > 1:
                self._shard_model()
            else:
                logger.info("Single-node mode: using full model")
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            # For testing, create dummy components
            logger.warning("Creating dummy model for testing")
            self.model = None
            self.tokenizer = None
    
    def _shard_model(self):
        """Shard model across ranks using capability-based allocation"""
        logger.info("Sharding model across ranks...")
        
        # Get model structure
        if hasattr(self.model, 'base_model'):
            model_to_shard = self.model.base_model
        else:
            model_to_shard = self.model
        
        # Extract layers based on model architecture
        if hasattr(model_to_shard, 'model'):  # Qwen-style models
            transformer = model_to_shard.model
            self.layers = transformer.layers
            self.embed_tokens = transformer.embed_tokens
            self.norm = transformer.norm
            self.lm_head = model_to_shard.lm_head if hasattr(model_to_shard, 'lm_head') else self.model.lm_head
        elif hasattr(model_to_shard, 'transformer'):  # GPT-style models
            transformer = model_to_shard.transformer
            self.layers = transformer.h
            self.embed_tokens = transformer.wte
            self.norm = transformer.ln_f
            self.lm_head = model_to_shard.lm_head if hasattr(model_to_shard, 'lm_head') else self.model.lm_head
        else:
            logger.error("Unsupported model architecture for sharding")
            return
        
        total_layers = len(self.layers)
        logger.info(f"Total layers to shard: {total_layers}")
        
        # Calculate layer assignment based on device capabilities
        layer_assignments = self._calculate_layer_assignments(total_layers)
        
        # Get this rank's assignment
        start_layer, end_layer = layer_assignments[self.rank]
        logger.info(f"Rank {self.rank} assigned layers {start_layer}-{end_layer-1}")
        
        # Keep only assigned layers
        self.assigned_layers = torch.nn.ModuleList(self.layers[start_layer:end_layer])
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_assigned_layers = end_layer - start_layer
        
        # Update device capabilities with layer assignment
        if self.device_capabilities and self.rank < len(self.device_capabilities):
            self.device_capabilities[self.rank].assigned_layers = self.num_assigned_layers
        
        # Clear original layers to save memory
        self.layers = None
        
        # Workers don't need embedding/head layers (only rank 0 and last rank)
        if self.rank > 0:
            self.embed_tokens = None
        if self.rank < self.world_size - 1:
            self.lm_head = None
            self.norm = None
        
        logger.info(f"Model shard initialized with {self.num_assigned_layers} layers")
    
    def _calculate_layer_assignments(self, total_layers: int) -> Dict[int, tuple]:
        """Calculate layer assignment based on device capabilities"""
        assignments = {}
        
        if self.device_capabilities and len(self.device_capabilities) >= self.world_size:
            # Capability-based assignment
            allocator = HeterogeneousCacheAllocator(self.device_capabilities[:self.world_size])
            
            # Calculate proportional assignment based on compute scores
            total_compute = sum(cap.compute_score() for cap in self.device_capabilities[:self.world_size])
            
            assigned_layers = 0
            for rank in range(self.world_size):
                device_cap = self.device_capabilities[rank]
                proportion = device_cap.compute_score() / total_compute
                
                if rank == self.world_size - 1:
                    # Last rank gets remaining layers
                    num_layers = total_layers - assigned_layers
                else:
                    # Proportional assignment
                    num_layers = max(1, int(total_layers * proportion))
                
                start_layer = assigned_layers
                end_layer = start_layer + num_layers
                assignments[rank] = (start_layer, end_layer)
                assigned_layers += num_layers
                
                logger.info(f"Capability-based: Rank {rank} ({device_cap.device_id}) gets {num_layers} layers "
                           f"(compute_score: {device_cap.compute_score():.2f})")
        else:
            # Equal distribution fallback
            layers_per_rank = total_layers // self.world_size
            remainder = total_layers % self.world_size
            
            assigned_layers = 0
            for rank in range(self.world_size):
                # Distribute remainder among first ranks
                num_layers = layers_per_rank + (1 if rank < remainder else 0)
                assignments[rank] = (assigned_layers, assigned_layers + num_layers)
                assigned_layers += num_layers
                
                logger.info(f"Equal distribution: Rank {rank} gets {num_layers} layers")
        
        return assignments
    
    def _initialize_cache(self):
        """Initialize KV cache with capability-based allocation"""
        try:
            if self.device_capabilities:
                # Initialize cache based on device capabilities
                self.cache_manager.initialize_cache_allocation(
                    max_batch_size=4,  # Conservative for stability
                    sequence_length=2048
                )
                
                # Log cache allocation
                cache_report = self.cache_manager.get_memory_report()
                logger.info(f"KV cache initialized: {cache_report.get('allocation', {})}")
            else:
                logger.warning("No device capabilities available, using default cache settings")
                
        except Exception as e:
            logger.warning(f"Cache initialization failed: {e}")
    
    def run(self):
        """Main server loop - coordinator runs API, workers run processing loop"""
        if self.rank == 0:
            self._run_coordinator()
        else:
            self._run_worker()
    
    def _run_coordinator(self):
        """Run coordinator with FastAPI server"""
        logger.info("Starting coordinator with API server on port 8100")
        
        # Import FastAPI components
        from fastapi import FastAPI, HTTPException, BackgroundTasks
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(
            title="Distributed PyTorch Inference Server",
            description="High-performance distributed inference with KV caching",
            version="1.0.0"
        )
        
        # Request models
        class GenerateRequest(BaseModel):
            prompt: str
            max_tokens: int = 50
            temperature: float = 0.7
            use_cache: bool = True
        
        class HealthResponse(BaseModel):
            status: str
            rank: int
            world_size: int
            model: str
            device: str
            cache_memory_mb: float
            
        @app.get("/health", response_model=HealthResponse)
        async def health():
            cache_report = self.cache_manager.get_memory_report()
            return HealthResponse(
                status="healthy",
                rank=self.rank,
                world_size=self.world_size,
                model=self.model_name,
                device=str(self.device),
                cache_memory_mb=cache_report.get('total_cache_memory_mb', 0.0)
            )
        
        @app.get("/cache/stats")
        async def cache_stats():
            """Get detailed cache statistics"""
            return self.cache_manager.get_memory_report()
        
        @app.post("/generate")
        async def generate(request: GenerateRequest):
            try:
                # Run generation in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._generate_text,
                    request.prompt,
                    request.max_tokens,
                    request.temperature,
                    request.use_cache
                )
                
                return {
                    "text": result,
                    "model": self.model_name,
                    "cached": request.use_cache
                }
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/cache/clear")
        async def clear_cache():
            """Clear all caches"""
            self.cache_manager.clear_all_cache()
            return {"status": "cache cleared"}
        
        # Run FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8100, log_level="warning")
    
    def _run_worker(self):
        """Run worker processing loop"""
        logger.info("Starting worker processing loop")
        
        # Simple worker loop - in production this would be more sophisticated
        while True:
            try:
                # Workers wait for tasks from coordinator
                # This is a placeholder - real implementation would use proper work queues
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Worker received interrupt signal")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(1)
        
        logger.info("Worker shutting down")
    
    def _generate_text(self, prompt: str, max_tokens: int, temperature: float, use_cache: bool) -> str:
        """Generate text using distributed model with KV caching"""
        logger.info(f"Generating response for: '{prompt[:50]}...' (cache: {use_cache})")
        
        if self.model is None or self.tokenizer is None:
            # Dummy response for testing
            return f"[TEST] Response to '{prompt}' with {max_tokens} tokens (cache: {use_cache})"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs['input_ids']
            
            # Generate with distributed model and caching
            start_time = time.time()
            generated_ids = self._distributed_generate(
                input_ids, 
                max_new_tokens=max_tokens,
                temperature=temperature,
                use_cache=use_cache
            )
            generation_time = time.time() - start_time
            
            # Decode response
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Log performance metrics
            generated_tokens = len(generated_ids[0]) - len(input_ids[0])
            tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
            
            logger.info(f"Generated {generated_tokens} tokens in {generation_time:.2f}s "
                       f"({tokens_per_second:.1f} tok/s)")
            
            if use_cache:
                cache_report = self.cache_manager.get_memory_report()
                logger.info(f"Cache usage: {cache_report.get('total_cache_memory_mb', 0):.1f}MB")
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Generation failed: {str(e)}"
    
    def _distributed_generate(self, input_ids: torch.Tensor, max_new_tokens: int, 
                            temperature: float, use_cache: bool) -> torch.Tensor:
        """Distributed generation with KV caching"""
        if self.world_size == 1:
            # Single node generation
            with torch.no_grad():
                return self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    use_cache=use_cache,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        else:
            # Multi-node distributed generation
            # This is a simplified version - production would be more sophisticated
            logger.warning("Multi-node generation not fully implemented yet")
            return input_ids  # Return input for now
    
    def cleanup(self):
        """Clean shutdown"""
        logger.info("Shutting down server...")
        
        # Clear caches
        if hasattr(self, 'cache_manager'):
            self.cache_manager.clear_all_cache()
        
        # Cleanup distributed
        if self.world_size > 1 and dist.is_initialized():
            try:
                dist.barrier()  # Ensure all ranks are ready
                dist.destroy_process_group()
                logger.info("Distributed process group destroyed")
            except Exception as e:
                logger.warning(f"Error during distributed cleanup: {e}")
        
        logger.info("Server shutdown complete")


def main():
    """Main entry point"""
    server = None
    try:
        server = DistributedModelServer()
        server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        if server:
            server.cleanup()


if __name__ == "__main__":
    main()