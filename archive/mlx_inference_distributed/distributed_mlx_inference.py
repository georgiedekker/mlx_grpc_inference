"""
Distributed MLX Inference Module

This module extends the single-node MLX inference to support distributed
inference across multiple devices using MPI for communication.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
from typing import List, Dict, Optional, Union, Callable, Tuple, Any
import logging
import numpy as np
from dataclasses import dataclass
import time

from distributed_config import DistributedConfig, DeviceRole
from distributed_comm import DistributedCommunicator, CommunicationType
from memory_efficient_model_loader import create_memory_efficient_model, MemoryEfficientShardedModel
from sharding_strategy import ResourceAwareShardingPlanner, ShardingStrategy
from device_capabilities import DeviceProfile

logger = logging.getLogger(__name__)


@dataclass
class ShardInfo:
    """Information about model sharding across devices."""
    total_layers: int
    layers_per_device: List[int]
    layer_assignments: Dict[int, int]  # layer_idx -> device_idx
    
    @classmethod
    def create_balanced(cls, total_layers: int, num_devices: int) -> "ShardInfo":
        """Create balanced layer distribution across devices."""
        base_layers = total_layers // num_devices
        extra_layers = total_layers % num_devices
        
        layers_per_device = [base_layers] * num_devices
        for i in range(extra_layers):
            layers_per_device[i] += 1
        
        layer_assignments = {}
        current_layer = 0
        for device_idx, num_layers in enumerate(layers_per_device):
            for _ in range(num_layers):
                layer_assignments[current_layer] = device_idx
                current_layer += 1
        
        return cls(total_layers, layers_per_device, layer_assignments)


class DistributedMLXInference:
    """
    Distributed MLX inference engine supporting model parallelism.
    
    This class handles:
    - Model sharding across multiple devices
    - Communication between devices during inference
    - Distributed generation with proper synchronization
    """
    
    def __init__(
        self, 
        config: DistributedConfig,
        communicator: DistributedCommunicator,
        local_rank: int
    ):
        """
        Initialize distributed inference engine.
        
        Args:
            config: Distributed system configuration
            communicator: Communication backend instance
            local_rank: Local device rank/index
        """
        self.config = config
        self.comm = communicator
        self.local_rank = local_rank
        self.device_config = config.get_device_by_index(local_rank)
        
        self.model = None
        self.tokenizer = None
        self.model_shard = None
        self.shard_info = None
        
        # Performance monitoring
        self.inference_times = []
        self.communication_times = []
        
        self._load_model()
    
    def _load_model(self):
        """Load and shard the model across devices using memory-efficient approach."""
        try:
            logger.info(f"Device {self.local_rank}: Loading model {self.config.model_name}")
            logger.info(f"Device {self.local_rank}: Default MLX device: {mx.default_device()}")
            
            # Create optimal sharding plan
            self._create_sharding_plan()
            
            # Load model using standard approach for now (debug)
            logger.info(f"Device {self.local_rank}: Loading model using standard approach")
            from model_abstraction import QwenModelWrapper
            wrapper = QwenModelWrapper(self.config.model_name)
            wrapper.load_model()  # Actually load the model
            self.model = wrapper.model
            self.tokenizer = wrapper.tokenizer
            
            # Ensure model is on GPU (MLX uses GPU by default on Apple Silicon)
            # Force a small computation to initialize GPU
            test_tensor = mx.ones((1, 1))
            mx.eval(test_tensor)
            
            # Only the master device keeps the tokenizer for generation
            if self.device_config.role != DeviceRole.MASTER:
                # Worker devices still need tokenizer for some operations, but we note the role
                pass
            
            # Get estimated memory usage (temporary fix)
            logger.info(f"Device {self.local_rank}: Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Device {self.local_rank}: Failed to load model: {str(e)}")
            raise
    
    def _create_sharding_plan(self):
        """Create optimal sharding plan for memory efficiency."""
        logger.info(f"Device {self.local_rank}: Creating sharding plan")
        
        # Create device profiles for all devices in the cluster
        devices = []
        for i, device_config in enumerate(self.config.device_list):
            # Convert config to DeviceProfile format
            device_profile = DeviceProfile(
                device_id=device_config.device_id,
                hostname=device_config.hostname,
                model=device_config.capabilities.get('model', 'Apple M4'),
                memory_gb=float(device_config.capabilities.get('memory_gb', 16.0)),
                gpu_cores=int(device_config.capabilities.get('gpu_cores', 10)),
                cpu_cores=int(device_config.capabilities.get('cpu_cores', 10)),
                cpu_performance_cores=int(device_config.capabilities.get('cpu_performance_cores', 4)),
                cpu_efficiency_cores=int(device_config.capabilities.get('cpu_efficiency_cores', 6)),
                neural_engine_cores=int(device_config.capabilities.get('neural_engine_cores', 16))
            )
            devices.append(device_profile)
        
        # Load model info to get layer count and parameters
        from memory_efficient_model_loader import MemoryEfficientModelLoader
        loader = MemoryEfficientModelLoader(self.config.model_name)
        model_info = loader.load_model_info_only()
        
        # Create sharding planner and find optimal strategy
        planner = ResourceAwareShardingPlanner()
        best_plan, all_plans = planner.find_optimal_strategy(model_info, devices)
        
        logger.info(f"Selected sharding strategy: {best_plan.strategy.value}")
        logger.info(f"Balance score: {best_plan.balance_score:.3f}")
        logger.info(f"Max memory utilization: {best_plan.max_memory_utilization:.1f}%")
        
        # Get this device's assignment
        self.shard_assignment = best_plan.assignments[self.local_rank]
        logger.info(f"Device {self.local_rank} assignment: layers {self.shard_assignment.start_layer}-{self.shard_assignment.end_layer-1}")
        
        # Store plan for reference
        self.sharding_plan = best_plan
    
    def _extract_model_shard(self):
        """Extract the appropriate model layers for this device."""
        # Determine which layers belong to this device
        my_layers = []
        for layer_idx, device_idx in self.shard_info.layer_assignments.items():
            if device_idx == self.local_rank:
                my_layers.append(layer_idx)
        
        logger.info(f"Device {self.local_rank}: Assigned layers {my_layers}")
        self.assigned_layers = set(my_layers)
        
        # For memory efficiency, we'll process only our layers
        # Check model structure
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Model has nested structure (like Qwen)
            self.model_layers = self.model.model.layers
            # Determine if we're using tied embeddings
            self.use_tied_embeddings = (hasattr(self.model, 'args') and 
                                       getattr(self.model.args, 'tie_word_embeddings', False))
            
            # First device gets embedding tokens, or last device if using tied embeddings
            if self.local_rank == 0:
                self.embed_tokens = self.model.model.embed_tokens
            elif self.local_rank == self.config.model_parallel_size - 1 and self.use_tied_embeddings:
                # Last device also needs embedding tokens for output projection
                self.embed_tokens = self.model.model.embed_tokens
            else:
                self.embed_tokens = None
                
            self.norm = self.model.model.norm if self.local_rank == self.config.model_parallel_size - 1 else None
            
            # Handle lm_head - check if it exists or if embeddings are tied
            if self.local_rank == self.config.model_parallel_size - 1:
                self.lm_head = getattr(self.model, 'lm_head', None)
            else:
                self.lm_head = None
        elif hasattr(self.model, 'layers'):
            # Direct layer access
            self.model_layers = self.model.layers
            # Determine if we're using tied embeddings
            self.use_tied_embeddings = (hasattr(self.model, 'args') and 
                                       getattr(self.model.args, 'tie_word_embeddings', False))
            
            # First device gets embedding tokens, or last device if using tied embeddings
            if self.local_rank == 0:
                self.embed_tokens = getattr(self.model, 'embed_tokens', None)
            elif self.local_rank == self.config.model_parallel_size - 1 and self.use_tied_embeddings:
                # Last device also needs embedding tokens for output projection
                self.embed_tokens = getattr(self.model, 'embed_tokens', None)
            else:
                self.embed_tokens = None
                
            self.norm = getattr(self.model, 'norm', None) if self.local_rank == self.config.model_parallel_size - 1 else None
            if self.local_rank == self.config.model_parallel_size - 1:
                self.lm_head = getattr(self.model, 'lm_head', None)
            else:
                self.lm_head = None
        else:
            logger.warning("Model doesn't have standard layer structure")
            self.model_shard = self.model
            return
        
        # Keep reference to full model for compatibility
        self.model_shard = self.model
    
    def _distributed_forward(
        self, 
        input_ids: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        """
        Perform distributed forward pass across all devices using memory-efficient shards.
        
        Args:
            input_ids: Input token IDs
            cache: Optional KV cache for attention layers
            
        Returns:
            Tuple of (logits, updated_cache)
        """
        start_time = time.time()
        
        # Use the memory-efficient sharded model
        if self.local_rank == 0:
            # First device: input_ids -> embeddings -> layers -> send to next
            logger.debug(f"Device 0: Processing input tokens shape: {input_ids.shape}")
            hidden_states = self.model(input_ids, cache)
            mx.eval(hidden_states)
            logger.debug(f"Device 0: Processed through assigned layers, output shape: {hidden_states.shape}")
        else:
            # Other devices: receive -> process layers -> send to next or return
            hidden_states = self.comm.receive(
                source=self.local_rank - 1,
                comm_type=CommunicationType.TENSOR
            )
            logger.debug(f"Device {self.local_rank}: Received hidden states shape: {hidden_states.shape}")
            
            # Process through our assigned layers
            hidden_states = self.model(hidden_states, cache)
            mx.eval(hidden_states)
            logger.debug(f"Device {self.local_rank}: Processed through assigned layers")
        
        # Handle communication between devices
        if self.local_rank < self.config.model_parallel_size - 1:
            # Send to next device
            self.comm.send(
                data=hidden_states,
                dest=self.local_rank + 1,
                comm_type=CommunicationType.TENSOR
            )
            logger.debug(f"Device {self.local_rank}: Sent hidden states to device {self.local_rank + 1}")
            
            # Wait for final logits from last device
            logits = self.comm.receive(
                source=self.config.model_parallel_size - 1,
                comm_type=CommunicationType.TENSOR
            )
            if logits is None:
                logger.error(f"Device {self.local_rank}: Received None logits")
                raise RuntimeError("Received None logits from last device")
            logger.debug(f"Device {self.local_rank}: Received final logits shape: {logits.shape}")
        else:
            # Last device: we have the final logits
            logits = hidden_states
            logger.debug(f"Device {self.local_rank}: Final logits shape: {logits.shape}")
            
            # Send logits back to all other devices
            for rank in range(self.config.model_parallel_size - 1):
                self.comm.send(
                    data=logits,
                    dest=rank,
                    comm_type=CommunicationType.TENSOR
                )
                logger.debug(f"Device {self.local_rank}: Sent logits to device {rank}")
        
        self.communication_times.append(time.time() - start_time)
        
        return logits, cache
    
    def generate_distributed(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        verbose: bool = False
    ) -> Tuple[str, int]:
        """
        Generate text using true distributed inference across devices.
        """
        start_time = time.time()
        
        # Skip barrier for now - workers aren't participating in the same flow
        logger.info(f"Device {self.local_rank}: Starting generation (barrier disabled)")
        
        if self.device_config.role == DeviceRole.MASTER:
            # Master coordinates the distributed generation
            if self.config.model_parallel_size == 1:
                # Single device fallback - use full model generation
                logger.info("Single device mode - using full model generation")
                sampler = make_sampler(temp=temperature, top_p=top_p)
                logits_processors = []
                if repetition_penalty > 1.0:
                    logits_processors.append(make_repetition_penalty(repetition_penalty))
                
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt_tokens,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors if logits_processors else None,
                    verbose=verbose
                )
                
                response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
                token_count = len(response_tokens)
                
                self.inference_times.append(time.time() - start_time)
                # self.comm.barrier()  # Disabled - workers not in same flow
                return response, token_count
            
            else:
                # True distributed generation
                logger.info("Multi-device mode - using distributed forward passes")
                
                # Convert prompt to tensor
                input_ids = mx.array(prompt_tokens).reshape(1, -1)
                generated_tokens = prompt_tokens.copy()
                
                sampler = make_sampler(temp=temperature, top_p=top_p)
                repetition_penalty_fn = None
                if repetition_penalty > 1.0:
                    repetition_penalty_fn = make_repetition_penalty(repetition_penalty)
                
                # Generate tokens one by one using distributed forward passes
                for step in range(max_tokens):
                    # Perform distributed forward pass
                    logits, _ = self._distributed_forward(input_ids)
                    
                    # Apply repetition penalty if specified
                    if repetition_penalty_fn is not None:
                        logits = repetition_penalty_fn(mx.array(generated_tokens), logits)
                    
                    # Sample next token
                    next_token = sampler(logits[:, -1:])
                    next_token_id = int(next_token[0, 0])
                    
                    # Check for end of sequence
                    if next_token_id == self.tokenizer.eos_token_id:
                        break
                    
                    generated_tokens.append(next_token_id)
                    
                    # Update input for next iteration
                    input_ids = mx.array([[next_token_id]])
                    
                    if verbose and step % 10 == 0:
                        logger.info(f"Generated {step} tokens")
                
                # Decode response
                response_tokens = generated_tokens[len(prompt_tokens):]
                response = self.tokenizer.decode(response_tokens)
                token_count = len(response_tokens)
                
                self.inference_times.append(time.time() - start_time)
                # self.comm.barrier()  # Disabled - workers not in same flow
                return response, token_count
        else:
            # Workers participate in distributed forward passes
            logger.info("Worker participating in distributed generation")
            
            if self.config.model_parallel_size == 1:
                # Single device - workers just wait
                # self.comm.barrier()  # Disabled - workers not in same flow
                return "", 0
            else:
                # Multi-device - workers participate in forward passes
                for step in range(max_tokens):
                    try:
                        # Workers receive input and participate in forward pass
                        # The _distributed_forward method handles worker communication
                        self._distributed_forward_worker()
                    except Exception as e:
                        logger.error(f"Worker {self.local_rank} error in step {step}: {e}")
                        break
                
                # self.comm.barrier()  # Disabled - workers not in same flow
                return "", 0
                
    def _distributed_forward_worker(self):
        """Worker participation in distributed forward pass."""
        # Workers receive hidden states, process their layers, and send to next device
        if self.local_rank > 0:
            # Receive hidden states from previous device
            hidden_states = self.comm.receive(
                source=self.local_rank - 1,
                comm_type=CommunicationType.TENSOR
            )
            
            # Process our assigned layers
            if hasattr(self, 'model_layers') and hasattr(self, 'assigned_layers'):
                for idx in sorted(self.assigned_layers):
                    if idx < len(self.model_layers):
                        layer = self.model_layers[idx]
                        if layer is not None:
                            hidden_states = layer(hidden_states)
                            mx.eval(hidden_states)
            
            # Send to next device or back to master
            if self.local_rank < self.config.model_parallel_size - 1:
                self.comm.send(
                    data=hidden_states,
                    dest=self.local_rank + 1,
                    comm_type=CommunicationType.TENSOR
                )
            else:
                # Last device computes final output and sends back to master
                if self.norm is not None:
                    hidden_states = self.norm(hidden_states)
                    mx.eval(hidden_states)
                
                if self.lm_head is not None:
                    logits = self.lm_head(hidden_states)
                    mx.eval(logits)
                elif self.use_tied_embeddings and self.embed_tokens is not None:
                    logits = self.embed_tokens.as_linear(hidden_states)
                    mx.eval(logits)
                else:
                    logits = hidden_states
                
                # Send logits back to master
                self.comm.send(
                    data=logits,
                    dest=0,
                    comm_type=CommunicationType.TENSOR
                )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        verbose: bool = False,
        return_token_count: bool = False
    ) -> Union[str, Tuple[str, int]]:
        """
        Generate a response in a chat conversation using distributed inference.
        
        Only the master device should call this method with actual messages.
        Worker devices should call with None messages.
        """
        logger.info(f"Device {self.local_rank}: chat() called, role={self.device_config.role}")
        
        if self.device_config.role == DeviceRole.MASTER:
            if not messages:
                raise ValueError("Messages list cannot be empty")
            
            # Format messages using chat template
            if self.tokenizer.chat_template is not None:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True
                )
            else:
                # Fallback formatting
                formatted_prompt = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in messages
                ])
                formatted_prompt += "\nassistant: "
            
            # Tokenize
            if isinstance(formatted_prompt, str):
                prompt_tokens = self.tokenizer.encode(formatted_prompt)
            else:
                prompt_tokens = formatted_prompt
        else:
            prompt_tokens = None
        
        # Generate response - all nodes participate through MPI collective operations
        response, token_count = self.generate_distributed(
            prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            verbose
        )
        
        if return_token_count:
            return response, token_count
        return response
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this device including memory savings."""
        stats = {
            "device_rank": self.local_rank,
            "device_id": self.device_config.device_id,
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0,
            "avg_communication_time": np.mean(self.communication_times) if self.communication_times else 0,
            "total_inferences": len(self.inference_times),
        }
        
        # Add memory efficiency stats
        if hasattr(self, 'shard_assignment') and hasattr(self, 'model'):
            stats.update({
                "assigned_layers": self.shard_assignment.layer_indices,
                "num_layers": len(self.shard_assignment.layer_indices),
                "actual_memory_usage_gb": self.model.get_memory_usage(),
                "estimated_memory_usage_gb": self.shard_assignment.estimated_memory_gb,
                "memory_saved_gb": self.shard_assignment.estimated_memory_gb - self.model.get_memory_usage(),
                "memory_utilization_percent": self.shard_assignment.memory_utilization(),
                "has_embedding": self.shard_assignment.has_embedding,
                "has_lm_head": self.shard_assignment.has_lm_head,
                "sharding_strategy": self.sharding_plan.strategy.value if hasattr(self, 'sharding_plan') else "unknown"
            })
        
        return stats