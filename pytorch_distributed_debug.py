#!/usr/bin/env python3
"""
PyTorch Distributed Server with Comprehensive Logging
Debug version to understand the distributed communication flow
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
import threading
import queue

# Setup logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - Rank %(rank)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class RankLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {**kwargs, 'extra': {'rank': self.extra.get('rank', '?')}}

logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': '?'})

class DistributedModelShard:
    """Manages a shard of the model on one device"""
    
    def __init__(self, model_name: str, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.model_name = model_name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Update logger
        global logger
        logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': rank})
        
        logger.info(f"=== Initializing model shard ===")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Rank: {rank}/{world_size}")
        
        # Load tokenizer (all ranks need this)
        logger.debug("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer loaded. Vocab size: {self.tokenizer.vocab_size}")
        
        # Load and shard model
        self._load_and_shard_model()
        
    def _load_and_shard_model(self):
        """Load model and keep only assigned layers"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Check if it's an MLX model
        if "mlx-community" in self.model_name:
            logger.info("Detected MLX model, using adapter...")
            # Use our MLX adapter
            from src.utils.mlx_pytorch_adapter import load_mlx_model_for_pytorch
            self.model, _ = load_mlx_model_for_pytorch(self.model_name, self.device)
            logger.info("MLX model loaded via adapter")
        else:
            # Load regular PyTorch model
            logger.info("Loading standard PyTorch model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
                device_map={"": self.device}
            )
        
        # Get model architecture info
        logger.debug("Analyzing model architecture...")
        if hasattr(self.model, 'base_model'):  # MLX adapter model
            logger.info("Found base_model (MLX adapter)")
            self.model = self.model.base_model
        
        if hasattr(self.model, 'model'):  # Qwen models
            logger.info("Detected Qwen architecture")
            transformer = self.model.model
            self.layers = transformer.layers
            self.embed_tokens = transformer.embed_tokens
            self.norm = transformer.norm
            self.lm_head = self.model.lm_head
            self.hidden_size = transformer.config.hidden_size
        elif hasattr(self.model, 'transformer'):  # GPT models
            logger.info("Detected GPT architecture")
            transformer = self.model.transformer
            self.layers = transformer.h
            self.embed_tokens = transformer.wte
            self.norm = transformer.ln_f
            self.lm_head = self.model.lm_head
            self.hidden_size = transformer.config.n_embd
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
        
        logger.info(f"Layer assignment: Total={total_layers}, Assigned={start_layer}-{end_layer-1}")
        logger.info(f"Hidden size: {self.hidden_size}")
        
        # Keep only assigned layers
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.assigned_layers = nn.ModuleList(self.layers[start_layer:end_layer])
        
        # Clear full model layers to save memory
        self.layers = None
        self.model = None
        
        logger.info(f"✓ Model shard initialized with {len(self.assigned_layers)} layers")
        
    def forward_shard(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through this shard's layers"""
        logger.debug(f"Forward shard: input shape={hidden_states.shape}")
        
        # Process through assigned layers
        for i, layer in enumerate(self.assigned_layers):
            layer_idx = self.start_layer + i
            logger.debug(f"Processing layer {layer_idx}")
            
            if hasattr(layer, 'forward'):
                # Handle different model architectures
                if attention_mask is not None:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
                else:
                    hidden_states = layer(hidden_states)[0]
            else:
                hidden_states = layer(hidden_states)
        
        logger.debug(f"Forward shard complete: output shape={hidden_states.shape}")
        return hidden_states

class DistributedInferenceEngine:
    """Coordinates distributed inference across model shards"""
    
    def __init__(self, model_name: str, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.model_shard = DistributedModelShard(model_name, rank, world_size)
        logger.info("=== Distributed Inference Engine initialized ===")
        
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, temperature: float = 0.7) -> torch.Tensor:
        """Generate text using distributed model"""
        if self.rank != 0:
            logger.info("Starting worker loop...")
            # Only rank 0 orchestrates generation
            self._worker_loop()
            return None
            
        # Rank 0: orchestrate generation
        logger.info("Starting generation orchestration...")
        return self._orchestrate_generation(input_ids, max_length, temperature)
    
    def _orchestrate_generation(self, input_ids: torch.Tensor, max_length: int, temperature: float) -> torch.Tensor:
        """Rank 0: Orchestrate the generation process"""
        logger.info(f"=== Starting generation ===")
        logger.info(f"Input shape: {input_ids.shape}")
        logger.info(f"Max length: {max_length}")
        logger.info(f"Temperature: {temperature}")
        
        device = self.model_shard.device
        generated = input_ids.to(device)
        
        for step in range(max_length):
            logger.debug(f"\n--- Generation step {step} ---")
            
            # Get model output for current sequence
            with torch.no_grad():
                # Step 1: Embedding layer (rank 0)
                logger.debug(f"Running embedding layer on sequence length {generated.shape[1]}")
                hidden_states = self.model_shard.embed_tokens(generated)
                logger.debug(f"Embedding output shape: {hidden_states.shape}")
                
                # Step 2: Forward through all shards
                hidden_states = self._distributed_forward(hidden_states)
                
                # Step 3: Get logits
                if self.world_size > 1:
                    logger.debug("Waiting for logits from distributed processing...")
                    # Receive logits from last rank
                    logits_shape = [hidden_states.shape[0], hidden_states.shape[1], self.model_shard.tokenizer.vocab_size]
                    logits_cpu = torch.zeros(logits_shape)
                    
                    logger.debug(f"Expecting logits shape: {logits_shape}")
                    dist.recv(logits_cpu, src=self.world_size-1)
                    logits = logits_cpu.to(device)
                    logger.debug(f"Received logits shape: {logits.shape}")
                else:
                    # Single node: do final processing here
                    logger.debug("Single node: applying norm and lm_head")
                    hidden_states = self.model_shard.norm(hidden_states)
                    logits = self.model_shard.lm_head(hidden_states)
                
                # Step 4: Sample next token
                next_token_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                logger.debug(f"Sampled token: {next_token.item()}")
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS
                if next_token.item() == self.model_shard.tokenizer.eos_token_id:
                    logger.info(f"EOS token generated at step {step}")
                    break
        
        # Signal workers to stop
        if self.world_size > 1:
            logger.info("Sending stop signal to workers...")
            stop_signal = torch.tensor([-1, -1, -1], dtype=torch.long)
            for rank in range(1, self.world_size):
                dist.send(stop_signal, dst=rank)
        
        logger.info(f"=== Generation complete. Total tokens: {generated.shape[1]} ===")
        return generated
    
    def _distributed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through all distributed shards"""
        logger.debug("Starting distributed forward pass")
        current_hidden = hidden_states
        
        # Process through rank 0's layers
        logger.debug(f"Processing rank 0 layers {self.model_shard.start_layer}-{self.model_shard.end_layer-1}")
        current_hidden = self.model_shard.forward_shard(current_hidden)
        logger.debug(f"Rank 0 processing complete. Shape: {current_hidden.shape}")
        
        # Send to next ranks and receive final result
        if self.world_size > 1:
            # Send to rank 1
            logger.debug(f"Sending tensor to rank 1. Shape: {current_hidden.shape}")
            
            # Send shape first
            shape_tensor = torch.tensor(current_hidden.shape, dtype=torch.long)
            dist.send(shape_tensor, dst=1)
            
            # Send data
            current_hidden_cpu = current_hidden.cpu()
            dist.send(current_hidden_cpu, dst=1)
            logger.debug("Sent to rank 1")
            
            # For 2-rank setup, also trigger final processing
            if self.world_size == 2:
                logger.debug("2-rank setup: Sending to rank 1 for final processing")
                # Send signal for final processing
                final_signal = torch.tensor(current_hidden.shape, dtype=torch.long)
                dist.send(final_signal, dst=1)
                dist.send(current_hidden_cpu, dst=1)
        
        return current_hidden
    
    def _worker_loop(self):
        """Worker ranks: process forwarded hidden states"""
        device = self.model_shard.device
        logger.info(f"=== Worker loop started for rank {self.rank} ===")
        
        while True:
            try:
                logger.debug(f"\n--- Worker rank {self.rank} waiting for data ---")
                
                # Step 1: Receive tensor shape
                shape_tensor = torch.zeros(3, dtype=torch.long)
                
                if self.rank == 1:
                    # Rank 1 receives from rank 0
                    logger.debug("Waiting for shape from rank 0...")
                    dist.recv(shape_tensor, src=0)
                else:
                    # Other ranks receive from previous rank
                    logger.debug(f"Waiting for shape from rank {self.rank-1}...")
                    dist.recv(shape_tensor, src=self.rank-1)
                
                logger.debug(f"Received shape tensor: {shape_tensor}")
                
                # Check for stop signal
                if shape_tensor[0].item() == -1:
                    logger.info(f"Rank {self.rank} received stop signal. Exiting worker loop.")
                    break
                
                # Step 2: Receive actual hidden states
                shape = tuple(shape_tensor.tolist())
                logger.debug(f"Expecting tensor of shape: {shape}")
                hidden_states_cpu = torch.zeros(shape)
                
                if self.rank == 1:
                    dist.recv(hidden_states_cpu, src=0)
                    logger.debug("Received hidden states from rank 0")
                else:
                    dist.recv(hidden_states_cpu, src=self.rank-1)
                    logger.debug(f"Received hidden states from rank {self.rank-1}")
                
                hidden_states = hidden_states_cpu.to(device)
                
                # Step 3: Process
                if self.rank == 1 and self.world_size == 2:
                    # In 2-rank setup, rank 1 does both layer processing AND final processing
                    
                    # First process through rank 1's layers
                    logger.debug(f"Processing layers {self.model_shard.start_layer}-{self.model_shard.end_layer-1}")
                    hidden_states = self.model_shard.forward_shard(hidden_states)
                    
                    # Then check if we need to do final processing
                    logger.debug("Checking for final processing signal...")
                    final_signal = torch.zeros(3, dtype=torch.long)
                    dist.recv(final_signal, src=0)
                    
                    if final_signal[0].item() != -1:
                        # Receive hidden states for final processing
                        final_shape = tuple(final_signal.tolist())
                        logger.debug(f"Final processing requested. Shape: {final_shape}")
                        
                        final_hidden_cpu = torch.zeros(final_shape)
                        dist.recv(final_hidden_cpu, src=0)
                        final_hidden = final_hidden_cpu.to(device)
                        
                        # Apply final norm and LM head
                        logger.debug("Applying norm and lm_head")
                        final_hidden = self.model_shard.norm(final_hidden)
                        logits = self.model_shard.lm_head(final_hidden)
                        logger.debug(f"Generated logits shape: {logits.shape}")
                        
                        # Send logits back to rank 0
                        logits_cpu = logits.cpu()
                        dist.send(logits_cpu, dst=0)
                        logger.debug("Sent logits to rank 0")
                elif self.rank < self.world_size - 1:
                    # Intermediate ranks: process and forward
                    logger.debug(f"Processing layers {self.model_shard.start_layer}-{self.model_shard.end_layer-1}")
                    hidden_states = self.model_shard.forward_shard(hidden_states)
                    
                    # Send to next rank
                    next_shape = torch.tensor(hidden_states.shape, dtype=torch.long)
                    dist.send(next_shape, dst=self.rank+1)
                    
                    hidden_states_cpu = hidden_states.cpu()
                    dist.send(hidden_states_cpu, dst=self.rank+1)
                    logger.debug(f"Forwarded to rank {self.rank+1}")
                else:
                    # Last rank (when world_size > 2): final processing
                    logger.debug(f"Last rank: processing layers {self.model_shard.start_layer}-{self.model_shard.end_layer-1}")
                    hidden_states = self.model_shard.forward_shard(hidden_states)
                    
                    # Apply final norm and LM head
                    logger.debug("Applying norm and lm_head")
                    hidden_states = self.model_shard.norm(hidden_states)
                    logits = self.model_shard.lm_head(hidden_states)
                    
                    # Send logits back to rank 0
                    logits_cpu = logits.cpu()
                    dist.send(logits_cpu, dst=0)
                    logger.debug("Sent logits to rank 0")
                    
            except Exception as e:
                logger.error(f"Worker loop error on rank {self.rank}: {e}", exc_info=True)
                break
        
        logger.info(f"=== Worker loop ended for rank {self.rank} ===")

def setup_distributed():
    """Initialize distributed environment with macOS workarounds"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Update logger
    global logger
    logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': rank})
    
    if world_size == 1:
        logger.info("Running in single-node mode")
        return rank, world_size
    
    # Master address configuration
    master_addr = os.environ.get('MASTER_ADDR', '192.168.5.1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    # IMPORTANT: Rank 0 binds to all interfaces
    if rank == 0:
        init_method = f'tcp://0.0.0.0:{master_port}'
    else:
        init_method = f'tcp://{master_addr}:{master_port}'
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    logger.info(f"=== Initializing distributed ===")
    logger.info(f"Rank: {rank}/{world_size}")
    logger.info(f"Master: {master_addr}:{master_port}")
    logger.info(f"Init method: {init_method}")
    logger.info(f"Hostname: {socket.gethostname()}")
    
    # Wait for master if worker
    if rank > 0:
        logger.info("Worker waiting 5s for master to start...")
        time.sleep(5)
    
    try:
        dist.init_process_group(
            backend='gloo',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=120)
        )
        logger.info("✓ Distributed initialization successful")
        
        # Test communication
        if rank == 0:
            logger.debug("Testing communication: sending test tensor to rank 1")
            test_tensor = torch.tensor([42.0])
            dist.send(test_tensor, dst=1)
            logger.debug("Test tensor sent")
        elif rank == 1:
            logger.debug("Testing communication: receiving test tensor from rank 0")
            test_tensor = torch.zeros(1)
            dist.recv(test_tensor, src=0)
            logger.debug(f"Test tensor received: {test_tensor.item()}")
            
    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}", exc_info=True)
        raise
    
    return rank, world_size

def main():
    """Main entry point"""
    try:
        # Setup distributed
        rank, world_size = setup_distributed()
        
        # Model configuration
        model_name = os.environ.get('MODEL_NAME', 'mlx-community/Qwen3-1.7B-8bit')
        
        # Initialize inference engine
        logger.info(f"Creating inference engine for {model_name}")
        engine = DistributedInferenceEngine(model_name, rank, world_size)
        
        if rank == 0:
            logger.info("=== Rank 0: Starting API server ===")
            
            # Import API server components
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            from typing import Dict, List, Optional, Union
            import uvicorn
            import time
            
            # Create simple API
            app = FastAPI(title="Distributed PyTorch Inference Debug")
            
            # Request queue for generation
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # Request models
            class CompletionRequest(BaseModel):
                model: str
                prompt: Union[str, List[str]]
                max_tokens: Optional[int] = 50
                temperature: Optional[float] = 0.7
            
            @app.get("/health")
            async def health():
                return {"status": "healthy", "rank": rank, "world_size": world_size}
            
            @app.post("/v1/completions")
            async def create_completion(request: CompletionRequest):
                logger.info(f"API: Received completion request")
                logger.info(f"Prompt: {request.prompt}")
                logger.info(f"Max tokens: {request.max_tokens}")
                
                try:
                    # Put request in queue
                    req_id = f"req-{int(time.time()*1000)}"
                    request_queue.put((req_id, request))
                    logger.debug(f"Request {req_id} queued")
                    
                    # Wait for response
                    start_time = time.time()
                    while True:
                        try:
                            resp_id, response = response_queue.get(timeout=30)
                            if resp_id == req_id:
                                logger.info(f"Request {req_id} completed in {time.time()-start_time:.2f}s")
                                if isinstance(response, Exception):
                                    raise response
                                return response
                        except queue.Empty:
                            logger.error("Request timeout")
                            raise HTTPException(status_code=504, detail="Request timeout")
                    
                except Exception as e:
                    logger.error(f"Error in completion: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Generation worker thread
            def generation_worker():
                """Worker thread that processes generation requests"""
                logger.info("Generation worker thread started")
                
                while True:
                    try:
                        req_id, request = request_queue.get(timeout=1)
                        logger.info(f"Generation worker: Processing request {req_id}")
                        
                        try:
                            # Tokenize
                            input_ids = engine.model_shard.tokenizer.encode(request.prompt, return_tensors='pt')
                            logger.debug(f"Tokenized input: {input_ids.shape}")
                            
                            # Generate
                            output_ids = engine.generate(
                                input_ids, 
                                max_length=request.max_tokens,
                                temperature=request.temperature
                            )
                            
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
                            logger.info(f"Request {req_id} response queued")
                            
                        except Exception as e:
                            logger.error(f"Generation error for {req_id}: {e}", exc_info=True)
                            response_queue.put((req_id, e))
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Generation worker error: {e}", exc_info=True)
            
            # Start generation worker thread
            gen_thread = threading.Thread(target=generation_worker, daemon=True)
            gen_thread.start()
            logger.info("Generation worker thread launched")
            
            # Run API server
            logger.info("Starting Uvicorn server on port 8100")
            uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
        else:
            logger.info(f"=== Rank {rank}: Entering worker loop ===")
            engine.generate(None, None, None)  # Enter worker loop
        
        # Cleanup
        if world_size > 1:
            logger.info("Starting cleanup...")
            dist.destroy_process_group()
        
        logger.info("✓ Shutting down gracefully")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()