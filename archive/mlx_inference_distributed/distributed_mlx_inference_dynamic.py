#!/usr/bin/env python3
"""
Enhanced distributed MLX inference with dynamic device support.
Preserves sophisticated features while fixing synchronization issues.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import threading
import queue

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as load_model
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.sample_utils import make_sampler

from distributed_config import DistributedConfig, DeviceRole
from distributed_comm import DistributedCommunicator, CommunicationType
from sharding_strategy import ShardInfo

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Request for distributed inference."""
    request_id: str
    input_ids: mx.array
    max_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float

@dataclass
class InferenceState:
    """State for ongoing inference."""
    request: InferenceRequest
    hidden_states: Optional[mx.array] = None
    generated_tokens: List[int] = None
    step: int = 0
    complete: bool = False

class DistributedMLXInferenceDynamic:
    """
    Enhanced distributed inference engine with dynamic device support.
    
    Key improvements:
    - No barriers or synchronized execution
    - Asynchronous tensor passing
    - Dynamic shard management
    - Partial model loading
    """
    
    def __init__(self, config: DistributedConfig, communicator: DistributedCommunicator):
        self.config = config
        self.comm = communicator
        self.local_rank = communicator.rank
        self.device_config = config.devices[self.local_rank]
        
        # Model components (loaded on demand)
        self.model = None
        self.tokenizer = None
        self.model_layers = None
        self.embed_tokens = None
        self.norm = None
        self.lm_head = None
        
        # Shard information
        self.shard_info: Optional[ShardInfo] = None
        self.assigned_layers = set()
        
        # Performance tracking
        self.inference_times = []
        self.communication_times = []
        
        # Dynamic state
        self._model_loaded = False
        self._shutdown = False
        
        # Inference queue for workers
        self._inference_queue = queue.Queue()
        self._results_queue = queue.Queue()
        
        # Start worker thread if not master
        if self.device_config.role == DeviceRole.WORKER:
            self._worker_thread = threading.Thread(target=self._worker_loop)
            self._worker_thread.daemon = True
            self._worker_thread.start()
        
        logger.info(f"Device {self.local_rank}: Initialized dynamic inference engine")
        
    def load_model_shard(self, shard_info: ShardInfo):
        """
        Load only the required model shard for this device.
        
        This is a key improvement - we only load what we need!
        """
        if self._model_loaded and self.shard_info == shard_info:
            logger.info(f"Device {self.local_rank}: Model shard already loaded")
            return
            
        self.shard_info = shard_info
        self.assigned_layers = set(range(shard_info.start_layer, shard_info.end_layer + 1))
        
        logger.info(f"Device {self.local_rank}: Loading model shard - layers {shard_info.start_layer}-{shard_info.end_layer}")
        
        # Load tokenizer (lightweight, okay to load on all devices)
        if not self.tokenizer:
            model_name = getattr(self.config, 'model_id', None) or getattr(self.config, 'model_name', None)
            # For now, let's skip the tokenizer and use the one from the model
            pass
        
        # Load full model structure (but we'll only keep our layers)
        if not self.model:
            model_name = getattr(self.config, 'model_id', None) or getattr(self.config, 'model_name', None)
            self.model, self.tokenizer = load_model(
                model_name,
                lazy=True  # Important: lazy loading to save memory
            )
        
        # Extract model structure
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            all_layers = self.model.model.layers
            
            # Only keep our assigned layers
            self.model_layers = {}
            for idx in self.assigned_layers:
                if idx < len(all_layers):
                    self.model_layers[idx] = all_layers[idx]
                    # Ensure layer is on GPU
                    mx.eval(self.model_layers[idx])
            
            # First device gets embeddings
            if shard_info.has_embeddings:
                self.embed_tokens = self.model.model.embed_tokens
                mx.eval(self.embed_tokens)
                logger.info(f"Device {self.local_rank}: Loaded embeddings")
            
            # Last device gets norm and lm_head
            if shard_info.has_head:
                self.norm = self.model.model.norm
                self.lm_head = getattr(self.model, 'lm_head', None)
                if self.norm:
                    mx.eval(self.norm)
                if self.lm_head:
                    mx.eval(self.lm_head)
                logger.info(f"Device {self.local_rank}: Loaded norm and lm_head")
                
            # Clear the full model to save memory
            self.model.model.layers = None
            
        self._model_loaded = True
        logger.info(f"Device {self.local_rank}: Model shard loaded successfully")
        
    def update_shard_assignment(self, new_shard_info: ShardInfo):
        """
        Dynamically update shard assignment when devices join/leave.
        """
        if self.shard_info != new_shard_info:
            logger.info(f"Device {self.local_rank}: Updating shard assignment")
            self.load_model_shard(new_shard_info)
            
    def _distributed_forward_async(self, input_ids: mx.array, request_id: str) -> mx.array:
        """
        Asynchronous distributed forward pass without barriers.
        """
        start_time = time.time()
        
        if self.device_config.role == DeviceRole.MASTER:
            # Master initiates the forward pass
            if self.embed_tokens is not None:
                hidden_states = self.embed_tokens(input_ids)
                mx.eval(hidden_states)
            else:
                # Receive from device that has embeddings
                hidden_states = self._receive_tensor_async(source=0)
                
            # Process our layers
            hidden_states = self._process_layers(hidden_states)
            
            # Send to next device if not last
            if self.local_rank < self.config.model_parallel_size - 1:
                self._send_tensor_async(hidden_states, dest=self.local_rank + 1, request_id=request_id)
                # Wait for final result
                logits = self._receive_tensor_async(source=self.config.model_parallel_size - 1)
            else:
                # We are the last device, compute logits
                logits = self._compute_logits(hidden_states)
                
            self.communication_times.append(time.time() - start_time)
            return logits
            
        else:
            # Worker devices handle requests asynchronously
            # This is handled in _worker_loop
            return None
            
    def _process_layers(self, hidden_states: mx.array) -> mx.array:
        """Process assigned transformer layers."""
        for idx in sorted(self.assigned_layers):
            if idx in self.model_layers:
                layer = self.model_layers[idx]
                hidden_states = layer(hidden_states)
                mx.eval(hidden_states)  # Force GPU computation
                
        return hidden_states
        
    def _compute_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute final logits (last device only)."""
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
            mx.eval(hidden_states)
            
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        elif hasattr(self, 'embed_tokens') and self.embed_tokens is not None:
            # Tied embeddings
            logits = self.embed_tokens.as_linear(hidden_states)
        else:
            logits = hidden_states
            
        mx.eval(logits)
        return logits
        
    def _send_tensor_async(self, tensor: mx.array, dest: int, request_id: str):
        """Send tensor asynchronously with request tracking."""
        # Tag with request_id for proper routing
        self.comm.send(
            data={"request_id": request_id, "tensor": tensor},
            dest=dest,
            comm_type=CommunicationType.PICKLE  # Use PICKLE for now to include metadata
        )
        
    def _receive_tensor_async(self, source: int, timeout: float = 30.0) -> mx.array:
        """Receive tensor asynchronously with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                data = self.comm.receive(source=source, comm_type=CommunicationType.PICKLE)
                if isinstance(data, dict) and "tensor" in data:
                    return data["tensor"]
                elif isinstance(data, mx.array):
                    return data
            except Exception as e:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout receiving tensor from {source}")
                time.sleep(0.01)
                
        raise TimeoutError(f"Timeout receiving tensor from {source}")
        
    def _worker_loop(self):
        """
        Worker loop that processes inference requests asynchronously.
        No barriers, no synchronization - just process requests as they come.
        """
        logger.info(f"Worker {self.local_rank}: Started worker loop")
        
        while not self._shutdown:
            try:
                # Check for incoming tensors
                if self.local_rank > 0:
                    # Try to receive from previous device
                    try:
                        data = self.comm.receive(
                            source=self.local_rank - 1,
                            comm_type=CommunicationType.PICKLE
                        )
                        
                        if isinstance(data, dict) and "tensor" in data:
                            request_id = data.get("request_id", "unknown")
                            hidden_states = data["tensor"]
                            
                            # Process our layers
                            hidden_states = self._process_layers(hidden_states)
                            
                            # Send to next device or back to master
                            if self.local_rank < self.config.model_parallel_size - 1:
                                # Send to next device
                                self._send_tensor_async(
                                    hidden_states,
                                    dest=self.local_rank + 1,
                                    request_id=request_id
                                )
                            else:
                                # Last device - compute logits and send back to master
                                logits = self._compute_logits(hidden_states)
                                self._send_tensor_async(logits, dest=0, request_id=request_id)
                                
                    except Exception as e:
                        # No data available, continue
                        time.sleep(0.01)
                        
            except Exception as e:
                if not self._shutdown:
                    logger.error(f"Worker {self.local_rank} error: {e}")
                    
    def generate_distributed(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 100,
        temp: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        verbose: bool = False
    ) -> Tuple[str, int]:
        """
        Generate text using distributed inference without barriers.
        """
        if not self._model_loaded:
            raise RuntimeError("Model shard not loaded. Call load_model_shard first.")
            
        start_time = time.time()
        
        # Only master coordinates generation
        if self.device_config.role != DeviceRole.MASTER:
            # Workers just process in their loop
            return "", 0
            
        # Convert prompt to tensor
        input_ids = mx.array(prompt_tokens).reshape(1, -1)
        
        # Generate tokens
        generated_tokens = prompt_tokens.copy()
        sampler = make_sampler(temp=temp, top_p=top_p)
        
        request_id = f"req_{int(time.time() * 1000)}"
        
        for step in range(max_tokens):
            # Distributed forward pass
            logits = self._distributed_forward_async(input_ids, request_id)
            
            if logits is None:
                logger.error("Received None logits")
                break
                
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                # Simple repetition penalty
                for token in set(generated_tokens):
                    if token < logits.shape[-1]:
                        logits[0, -1, token] /= repetition_penalty
                        
            # Sample next token
            next_token = sampler(logits[:, -1:])
            next_token_id = int(next_token.item())
            
            # Check for stop
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id)
            input_ids = mx.array([[next_token_id]])
            
            if verbose:
                print(self.tokenizer.decode([next_token_id]), end="", flush=True)
                
        # Decode response
        response_tokens = generated_tokens[len(prompt_tokens):]
        response = self.tokenizer.decode(response_tokens)
        
        self.inference_times.append(time.time() - start_time)
        
        # Calculate tokens per second
        total_time = time.time() - start_time
        tps = len(response_tokens) / total_time if total_time > 0 else 0
        logger.info(f"Generated {len(response_tokens)} tokens in {total_time:.2f}s ({tps:.1f} tokens/sec)")
        
        return response, len(response_tokens)
        
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        return_token_count: bool = False
    ) -> Union[str, Tuple[str, int]]:
        """
        Chat interface compatible with OpenAI API.
        """
        # Format messages into prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
                
        prompt += "Assistant: "
        
        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt)
        
        # Generate response
        response, token_count = self.generate_distributed(
            prompt_tokens,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        if return_token_count:
            return response, token_count
        return response
        
    def shutdown(self):
        """Clean shutdown of inference engine."""
        logger.info(f"Device {self.local_rank}: Shutting down inference engine")
        self._shutdown = True
        if hasattr(self, '_worker_thread'):
            self._worker_thread.join(timeout=5.0)
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of model shard."""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_gb": memory_info.rss / (1024**3),
            "vms_gb": memory_info.vms / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "percent": psutil.virtual_memory().percent
        }