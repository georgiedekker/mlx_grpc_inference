#!/usr/bin/env python3
"""
PyTorch Distributed Server with Model Sharding
Works around macOS Gloo issues by using explicit network configuration
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket
import datetime
import logging
import asyncio
from typing import Optional, Tuple, List
import time
import json

# Import our KV cache implementation
from src.core.pytorch_kv_cache import (
    DistributedKVCacheManager, 
    DeviceCapability, 
    load_device_capabilities_from_config,
    estimate_kv_cache_memory
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Rank %(rank)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class RankLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {**kwargs, 'extra': {'rank': self.extra.get('rank', '?')}}

logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': '?'})

class DistributedModelShard:
    """Manages a shard of the model on one device with KV caching"""
    
    def __init__(self, model_name: str, rank: int, world_size: int, device_capabilities: Optional[List[DeviceCapability]] = None):
        self.rank = rank
        self.world_size = world_size
        self.model_name = model_name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Update logger
        global logger
        logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': rank})
        
        logger.info(f"Initializing model shard on {self.device}")
        
        # Initialize KV cache manager
        self.device_capabilities = device_capabilities
        self.cache_manager = DistributedKVCacheManager(
            rank=rank,
            world_size=world_size,
            device=self.device,
            device_capabilities=device_capabilities
        )
        
        # Load tokenizer (all ranks need this)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and shard model
        self._load_and_shard_model()
        
        # Initialize cache after model loading
        self._initialize_cache()
        
    def _load_and_shard_model(self):
        """Load model and keep only assigned layers"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Check if it's an MLX model
        if "mlx-community" in self.model_name:
            # Use our MLX adapter
            from src.utils.mlx_pytorch_adapter import load_mlx_model_for_pytorch
            self.model, _ = load_mlx_model_for_pytorch(self.model_name, self.device)
        else:
            # Load regular PyTorch model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
                device_map={"": self.device}
            )
        
        # Get model architecture info
        if hasattr(self.model, 'base_model'):  # MLX adapter model
            self.model = self.model.base_model
        
        if hasattr(self.model, 'model'):  # Qwen models
            transformer = self.model.model
            self.layers = transformer.layers
            self.embed_tokens = transformer.embed_tokens
            self.norm = transformer.norm
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, 'transformer'):  # GPT models
            transformer = self.model.transformer
            self.layers = transformer.h
            self.embed_tokens = transformer.wte
            self.norm = transformer.ln_f
            self.lm_head = self.model.lm_head
        else:
            raise ValueError("Unsupported model architecture")
        
        total_layers = len(self.layers)
        layers_per_rank = total_layers // self.world_size
        remainder = total_layers % self.world_size
        
        # Calculate layer assignment
        if self.rank < remainder:
            start_layer = self.rank * (layers_per_rank + 1)
            end_layer = start_layer + layers_per_rank + 1
        else:
            start_layer = self.rank * layers_per_rank + remainder
            end_layer = start_layer + layers_per_rank
        
        logger.info(f"Total layers: {total_layers}, assigned layers: {start_layer}-{end_layer-1}")
        
        # Keep only assigned layers
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.assigned_layers = nn.ModuleList(self.layers[start_layer:end_layer])
        
        # Clear full model layers to save memory
        self.layers = None
        self.model = None
        
        logger.info(f"Model shard initialized with {len(self.assigned_layers)} layers")
        
        # Store layer assignment info for cache allocation
        if self.device_capabilities and self.rank < len(self.device_capabilities):
            self.device_capabilities[self.rank].assigned_layers = len(self.assigned_layers)
        
    def _initialize_cache(self):
        """Initialize KV cache based on device capabilities"""
        try:
            # Estimate memory requirements
            model_config = {
                'num_attention_heads': getattr(self.model.config if hasattr(self, 'model') else self.tokenizer, 'num_attention_heads', 16),
                'hidden_size': getattr(self.model.config if hasattr(self, 'model') else self.tokenizer, 'hidden_size', 2048)
            }
            
            # Initialize cache allocation
            self.cache_manager.initialize_cache_allocation(
                max_batch_size=4,  # Conservative batch size
                sequence_length=2048
            )
            
            logger.info(f"KV cache initialized for rank {self.rank}")
            
        except Exception as e:
            logger.warning(f"Could not initialize KV cache: {e}")
    
    def forward_shard(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                     use_cache: bool = True, past_key_values: Optional[List] = None, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[List]]:
        """Forward pass through this shard's layers with KV caching"""
        new_key_values = [] if use_cache else None
        
        # Process through assigned layers
        for i, layer in enumerate(self.assigned_layers):
            layer_idx = self.start_layer + i
            
            # Get past key values for this layer if available
            past_kv = None
            if past_key_values and len(past_key_values) > i:
                past_kv = past_key_values[i]
            elif use_cache:
                # Try to get from cache manager
                cached_kv = self.cache_manager.get_cached_kv(layer_idx, 0)  # Simplified for now
                if cached_kv[0] is not None:
                    past_kv = cached_kv
            
            # Forward through layer
            if hasattr(layer, 'forward'):
                # Handle different model architectures with cache support
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache
                )
                
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                    if use_cache and len(layer_outputs) > 1:
                        new_kv = layer_outputs[1]
                        new_key_values.append(new_kv)
                        
                        # Update cache manager
                        if new_kv is not None and len(new_kv) == 2:
                            self.cache_manager.update_cache(layer_idx, new_kv[0], new_kv[1], 0)
                else:
                    hidden_states = layer_outputs
            else:
                hidden_states = layer(hidden_states)
                
        return hidden_states, new_key_values if use_cache else None

class DistributedInferenceEngine:
    """Coordinates distributed inference across model shards with KV caching"""
    
    def __init__(self, model_name: str, rank: int, world_size: int, config_path: Optional[str] = None):
        self.rank = rank
        self.world_size = world_size
        
        # Load device capabilities if config provided
        self.device_capabilities = None
        if config_path:
            try:
                self.device_capabilities = load_device_capabilities_from_config(config_path)
                logger.info(f"Loaded capabilities for {len(self.device_capabilities)} devices")
            except Exception as e:
                logger.warning(f"Could not load device capabilities: {e}")
        
        self.model_shard = DistributedModelShard(model_name, rank, world_size, self.device_capabilities)
        
        # Track generation state for caching
        self.generation_cache = {}
        self.sequence_counter = 0
        
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, temperature: float = 0.7, use_cache: bool = True) -> torch.Tensor:
        """Generate text using distributed model with KV caching"""
        if self.rank != 0:
            # Only rank 0 orchestrates generation
            self._worker_loop()
            return None
            
        # Rank 0: orchestrate generation
        return self._orchestrate_generation(input_ids, max_length, temperature, use_cache)
    
    def _orchestrate_generation(self, input_ids: torch.Tensor, max_length: int, temperature: float, use_cache: bool = True) -> torch.Tensor:
        """Rank 0: Orchestrate the generation process with KV caching"""
        device = self.model_shard.device
        generated = input_ids.to(device)
        
        # Initialize generation state
        self.sequence_counter += 1
        seq_id = f"seq_{self.sequence_counter}"
        past_key_values = None
        
        for step in range(max_length):
            # Get model output for current sequence
            with torch.no_grad():
                # For cached generation, only process new tokens after first step
                if use_cache and step > 0:
                    # Only process the last token
                    input_for_step = generated[:, -1:]
                else:
                    # Process full sequence for first step
                    input_for_step = generated
                
                # Embedding layer (rank 0)
                hidden_states = self.model_shard.embed_tokens(input_for_step)
                
                # Forward through all shards with cache
                hidden_states, new_past_key_values = self._distributed_forward_with_cache(
                    hidden_states, use_cache=use_cache, past_key_values=past_key_values
                )
                
                if use_cache:
                    past_key_values = new_past_key_values
                
                # Final norm and LM head (on last rank, need to receive)
                if self.world_size > 1:
                    # Send to last rank for final processing
                    hidden_states_cpu = hidden_states.cpu()
                    dist.send(hidden_states_cpu, dst=self.world_size-1)
                    
                    # Receive logits back
                    logits_shape = (hidden_states.shape[0], hidden_states.shape[1], self.model_shard.tokenizer.vocab_size)
                    logits_cpu = torch.zeros(logits_shape)
                    dist.recv(logits_cpu, src=self.world_size-1)
                    logits = logits_cpu.to(device)
                else:
                    # Single node: do final processing here
                    hidden_states = self.model_shard.norm(hidden_states)
                    logits = self.model_shard.lm_head(hidden_states)
                
                # Sample next token
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS
                if next_token.item() == self.model_shard.tokenizer.eos_token_id:
                    break
        
        # Log cache usage
        if use_cache:
            cache_report = self.model_shard.cache_manager.get_memory_report()
            logger.info(f"Generation complete. Cache usage: {cache_report['total_cache_memory_mb']:.1f}MB")
        
        # Signal workers to stop
        if self.world_size > 1:
            stop_signal = torch.tensor([-1.0])
            for rank in range(1, self.world_size):
                dist.send(stop_signal, dst=rank)
        
        return generated
    
    def _distributed_forward_with_cache(self, hidden_states: torch.Tensor, use_cache: bool = True, 
                                       past_key_values: Optional[List] = None) -> Tuple[torch.Tensor, Optional[List]]:
        """Forward pass through all distributed shards with KV caching"""
        current_hidden = hidden_states
        all_new_key_values = []
        
        # Process through rank 0's layers
        current_hidden, new_kv = self.model_shard.forward_shard(
            current_hidden, use_cache=use_cache, past_key_values=past_key_values
        )
        
        if use_cache and new_kv:
            all_new_key_values.extend(new_kv)
        
        # Send to next ranks and receive final result
        if self.world_size > 1:
            # Send shape first to rank 1
            shape_tensor = torch.tensor(current_hidden.shape, dtype=torch.long)
            dist.send(shape_tensor, dst=1)
            
            # Send data to rank 1
            current_hidden_cpu = current_hidden.cpu()
            dist.send(current_hidden_cpu, dst=1)
            
            # For 2-rank setup, also send to last rank for final processing
            if self.world_size == 2:
                # Send shape and data to rank 1 for final processing
                final_shape = torch.tensor(current_hidden.shape, dtype=torch.long)
                dist.send(final_shape, dst=1)
                dist.send(current_hidden_cpu, dst=1)
            
            # Wait for result from pipeline
            if self.world_size > 2:
                # Receive from rank 1 (which forwards from last rank)
                final_shape = torch.zeros(3, dtype=torch.long)
                dist.recv(final_shape, src=1)
                
                final_hidden_cpu = torch.zeros(tuple(final_shape.tolist()))
                dist.recv(final_hidden_cpu, src=1)
                current_hidden = final_hidden_cpu.to(hidden_states.device)
            else:
                # Two ranks: receive directly from rank 1
                # Receive logits shape
                logits_shape = torch.zeros(3, dtype=torch.long)
                dist.recv(logits_shape, src=1)
                
                # Return dummy for logits (will be received separately)
                return None, all_new_key_values if use_cache else None
        
        return current_hidden, all_new_key_values if use_cache else None
    
    def _distributed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Backward compatibility wrapper for non-cached forward"""
        result, _ = self._distributed_forward_with_cache(hidden_states, use_cache=False)
        return result
    
    def _worker_loop(self):
        """Worker ranks: process forwarded hidden states"""
        device = self.model_shard.device
        logger.info(f"Worker loop started for rank {self.rank}")
        
        while True:
            try:
                # Receive tensor shape first
                shape_tensor = torch.zeros(3, dtype=torch.long)  # [batch, seq_len, hidden_dim]
                
                if self.rank == self.world_size - 1:
                    # Last rank receives from rank 0 for final processing
                    dist.recv(shape_tensor, src=0)
                else:
                    # Other ranks receive from previous rank
                    dist.recv(shape_tensor, src=self.rank-1)
                
                # Check for stop signal
                if shape_tensor[0].item() == -1:
                    logger.info(f"Rank {self.rank} received stop signal")
                    break
                
                # Receive actual hidden states
                shape = tuple(shape_tensor.tolist())
                hidden_states_cpu = torch.zeros(shape)
                
                if self.rank == self.world_size - 1:
                    dist.recv(hidden_states_cpu, src=0)
                else:
                    dist.recv(hidden_states_cpu, src=self.rank-1)
                
                hidden_states = hidden_states_cpu.to(device)
                
                # Last rank does final processing
                if self.rank == self.world_size - 1:
                    # Apply final norm and LM head
                    hidden_states = self.model_shard.norm(hidden_states)
                    logits = self.model_shard.lm_head(hidden_states)
                    
                    # Send logits back to rank 0
                    logits_cpu = logits.cpu()
                    dist.send(logits_cpu, dst=0)
                else:
                    # Process through this shard's layers
                    hidden_states = self.model_shard.forward_shard(hidden_states)
                    
                    # Send to next rank
                    hidden_states_cpu = hidden_states.cpu()
                    
                    # Send shape first
                    next_shape = torch.tensor(hidden_states.shape, dtype=torch.long)
                    dist.send(next_shape, dst=self.rank+1)
                    
                    # Send data
                    dist.send(hidden_states_cpu, dst=self.rank+1)
                    
                    # If this is rank 1 and world_size > 2, also receive from last rank and forward to rank 0
                    if self.rank == 1 and self.world_size > 2:
                        # Pipeline: receive final result from last rank
                        final_shape = torch.zeros(3, dtype=torch.long)
                        dist.recv(final_shape, src=self.world_size-1)
                        
                        if final_shape[0].item() != -1:
                            final_hidden = torch.zeros(tuple(final_shape.tolist()))
                            dist.recv(final_hidden, src=self.world_size-1)
                            
                            # Forward to rank 0
                            dist.send(final_shape, dst=0)
                            dist.send(final_hidden, dst=0)
                    
            except Exception as e:
                logger.error(f"Worker loop error on rank {self.rank}: {e}")
                break

def setup_distributed():
    """Initialize distributed environment with macOS workarounds"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size == 1:
        logger.info("Running in single-node mode")
        return rank, world_size
    
    # Master address configuration
    master_addr = os.environ.get('MASTER_ADDR', '192.168.5.1')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    # IMPORTANT: Rank 0 binds to all interfaces
    if rank == 0:
        init_method = f'tcp://0.0.0.0:{master_port}'
    else:
        init_method = f'tcp://{master_addr}:{master_port}'
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    logger.info(f"Initializing distributed: rank={rank}, world_size={world_size}, init_method={init_method}")
    
    # Wait for master if worker
    if rank > 0:
        time.sleep(5)  # Give master time to bind
    
    try:
        dist.init_process_group(
            backend='gloo',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=120)
        )
        logger.info("Distributed initialization successful")
    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}")
        raise
    
    return rank, world_size

def main():
    """Main entry point"""
    try:
        # Setup distributed
        rank, world_size = setup_distributed()
        
        # Model configuration
        model_name = os.environ.get('MODEL_NAME', 'microsoft/phi-2')  # Smaller model for testing
        
        # Initialize inference engine with config
        logger.info(f"Creating inference engine for {model_name}")
        config_path = "config/cluster_config.yaml" if os.path.exists("config/cluster_config.yaml") else None
        engine = DistributedInferenceEngine(model_name, rank, world_size, config_path)
        
        if rank == 0:
            logger.info("Rank 0: Starting API server on port 8100")
            
            # Import API server components
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            from typing import Dict, List, Optional, Union
            import uvicorn
            import time
            import threading
            import queue
            
            # Create simple API
            app = FastAPI(title="Distributed PyTorch Inference")
            
            # Request queue for generation
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # Request models
            class CompletionRequest(BaseModel):
                model: str
                prompt: Union[str, List[str]]
                max_tokens: Optional[int] = 100
                temperature: Optional[float] = 0.7
            
            @app.get("/health")
            async def health():
                return {"status": "healthy", "rank": rank, "world_size": world_size}
            
            @app.post("/v1/completions")
            async def create_completion(request: CompletionRequest):
                try:
                    # Put request in queue
                    req_id = f"req-{int(time.time()*1000)}"
                    request_queue.put((req_id, request))
                    
                    # Wait for response
                    while True:
                        try:
                            resp_id, response = response_queue.get(timeout=30)
                            if resp_id == req_id:
                                if isinstance(response, Exception):
                                    raise response
                                return response
                        except queue.Empty:
                            raise HTTPException(status_code=504, detail="Request timeout")
                    
                except Exception as e:
                    logger.error(f"Error in completion: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Generation worker thread
            def generation_worker():
                """Worker thread that processes generation requests"""
                while True:
                    try:
                        req_id, request = request_queue.get(timeout=1)
                        
                        # Tokenize
                        input_ids = engine.model_shard.tokenizer.encode(request.prompt, return_tensors='pt')
                        
                        # Generate
                        output_ids = engine.generate(input_ids, max_length=len(input_ids[0]) + request.max_tokens, temperature=request.temperature)
                        
                        # Decode
                        generated_text = engine.model_shard.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        # Remove prompt from output
                        if generated_text.startswith(request.prompt):
                            generated_text = generated_text[len(request.prompt):]
                        
                        response = {
                            "id": f"cmpl-{int(time.time()*1000)}",
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "text": generated_text,
                                "index": 0,
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": len(input_ids[0]),
                                "completion_tokens": len(output_ids[0]) - len(input_ids[0]),
                                "total_tokens": len(output_ids[0])
                            }
                        }
                        
                        response_queue.put((req_id, response))
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Generation error: {e}")
                        response_queue.put((req_id, e))
            
            # Start generation worker thread
            gen_thread = threading.Thread(target=generation_worker, daemon=True)
            gen_thread.start()
            
            # Run API server
            uvicorn.run(app, host="0.0.0.0", port=8100, log_level="warning")
        else:
            logger.info(f"Rank {rank}: Entering worker loop")
            engine.generate(None, None, None)  # Enter worker loop
        
        # Cleanup
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        
        logger.info("Shutting down gracefully")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()