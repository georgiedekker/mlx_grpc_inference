#!/usr/bin/env python3
"""
Distributed training implementation for MLX with proper tensor serialization
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_flatten, tree_unflatten
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing import Queue, Process
import os
import time
import json
import logging
from pathlib import Path

# Import the new distributed communication components
from .distributed_comm import (
    DistributedCommunicator,
    serialize_mlx_tensors,
    deserialize_mlx_tensors,
    prepare_gradients_for_allreduce,
    restore_gradients_from_allreduce
)
from .grpc_server import TensorSerializer, create_distributed_server

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    world_size: int = 1
    rank: int = 0
    backend: str = "allreduce"  # allreduce, ring_allreduce, parameter_server
    master_addr: str = "localhost"
    master_port: int = 29500
    gradient_as_bucket_view: bool = True
    find_unused_parameters: bool = False
    devices: List[str] = None  # List of device identifiers (e.g., ["gpu:0", "gpu:1"])
    
    # Communication configuration
    timeout: int = 30000  # Communication timeout in milliseconds
    retry_count: int = 3  # Number of retries for failed operations
    
    # Optimization settings
    overlap_grad_reduce: bool = True  # Overlap gradient reduction with computation
    bucket_size_mb: int = 25  # Gradient bucket size in MB
    
    def __post_init__(self):
        """Initialize default devices if not provided."""
        if self.devices is None:
            self.devices = [f"gpu:{i}" for i in range(self.world_size)]
        elif len(self.devices) != self.world_size:
            raise ValueError(f"Number of devices ({len(self.devices)}) must match world_size ({self.world_size})")
    
    @classmethod
    def from_json(cls, json_path: str) -> 'DistributedConfig':
        """Load configuration from JSON file."""
        import json
        with open(json_path, 'r') as f:
            config_data = json.load(f)
        
        # Handle nested configuration
        if 'communication' in config_data:
            comm_config = config_data.pop('communication')
            config_data.update(comm_config)
        
        if 'optimization' in config_data:
            opt_config = config_data.pop('optimization')
            config_data.update(opt_config)
        
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "backend": self.backend,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "gradient_as_bucket_view": self.gradient_as_bucket_view,
            "find_unused_parameters": self.find_unused_parameters,
            "devices": self.devices,
            "communication": {
                "timeout": self.timeout,
                "retry_count": self.retry_count
            },
            "optimization": {
                "overlap_grad_reduce": self.overlap_grad_reduce,
                "bucket_size_mb": self.bucket_size_mb
            }
        }
        

class AllReduceBackend:
    """All-reduce communication backend with proper MLX tensor serialization."""
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.comm_group = None
        self.communicator = DistributedCommunicator(rank, world_size)
        self.tensor_serializer = TensorSerializer()
        
    def init_process_group(self):
        """Initialize process group for communication."""
        # In real implementation, this would set up MPI or NCCL
        # For now, we'll use multiprocessing queues
        logger.info(f"Initializing process group for rank {self.rank}/{self.world_size}")
        pass
        
    def all_reduce(self, tensor: mx.array, op: str = "sum") -> mx.array:
        """All-reduce operation across all processes with proper serialization."""
        # Simplified all-reduce (in practice, use MPI/NCCL)
        if self.world_size == 1:
            return tensor
        
        try:
            # Validate tensor compatibility before communication
            if not self.tensor_serializer.validate_tensor_compatibility(tensor):
                logger.warning(f"Fixing tensor compatibility issues for all-reduce")
                tensor = self.tensor_serializer.fix_tensor_dtype_issues(tensor)
            
            # Use the communicator for distributed operation
            tensor_dict = {"tensor": tensor}
            reduced_dict = self.communicator.all_reduce(tensor_dict, op)
            reduced_tensor = reduced_dict["tensor"]
            
            logger.debug(f"All-reduce completed for tensor with shape {tensor.shape}, dtype {tensor.dtype}")
            return reduced_tensor
            
        except Exception as e:
            logger.error(f"All-reduce failed: {e}")
            # Fallback to local operation
            if op == "mean":
                return tensor / self.world_size
            return tensor
        
    def broadcast(self, tensor: mx.array, src: int = 0) -> mx.array:
        """Broadcast tensor from source to all processes with proper serialization."""
        if self.world_size == 1:
            return tensor
        
        try:
            # Validate tensor compatibility before communication
            if not self.tensor_serializer.validate_tensor_compatibility(tensor):
                logger.warning(f"Fixing tensor compatibility issues for broadcast")
                tensor = self.tensor_serializer.fix_tensor_dtype_issues(tensor)
            
            # Use the communicator for broadcast operation
            tensor_dict = {"tensor": tensor}
            broadcast_dict = self.communicator.broadcast(tensor_dict, src)
            broadcast_tensor = broadcast_dict["tensor"]
            
            logger.debug(f"Broadcast completed for tensor with shape {tensor.shape}, dtype {tensor.dtype}")
            return broadcast_tensor
            
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
            # Fallback to returning original tensor
            return tensor
        
    def barrier(self):
        """Synchronization barrier."""
        # Wait for all processes
        pass
        

class RingAllReduceBackend:
    """Ring all-reduce for efficient gradient aggregation."""
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.next_rank = (rank + 1) % world_size
        self.prev_rank = (rank - 1) % world_size
        
    def ring_all_reduce(self, tensor: mx.array) -> mx.array:
        """Ring all-reduce implementation."""
        if self.world_size == 1:
            return tensor
            
        # Ring reduce-scatter
        chunk_size = tensor.size // self.world_size
        chunks = mlx.split(tensor.flatten(), self.world_size)
        
        # Each process reduces its chunk
        for i in range(self.world_size - 1):
            send_idx = (self.rank - i) % self.world_size
            recv_idx = (self.rank - i - 1) % self.world_size
            
            # Send chunks[send_idx] to next, receive into chunks[recv_idx]
            # In practice, use MPI send/recv
            pass
            
        # Ring all-gather
        for i in range(self.world_size - 1):
            send_idx = (self.rank - i + 1) % self.world_size
            recv_idx = (self.rank - i) % self.world_size
            
            # Send chunks[send_idx] to next, receive into chunks[recv_idx]
            pass
            
        # Reshape back
        return mlx.concatenate(chunks).reshape(tensor.shape)
        

class ParameterServerBackend:
    """Parameter server for asynchronous training."""
    
    def __init__(self, world_size: int, rank: int, is_server: bool = False):
        self.world_size = world_size
        self.rank = rank
        self.is_server = is_server
        self.parameter_cache = {}
        
    def push_gradients(self, gradients: Dict[str, mx.array]):
        """Push gradients to parameter server."""
        if self.is_server:
            # Aggregate gradients
            for name, grad in gradients.items():
                if name in self.parameter_cache:
                    self.parameter_cache[name] += grad
                else:
                    self.parameter_cache[name] = grad
        else:
            # Send to server
            pass
            
    def pull_parameters(self) -> Dict[str, mx.array]:
        """Pull latest parameters from server."""
        if self.is_server:
            # Return aggregated parameters
            params = {}
            for name, grad_sum in self.parameter_cache.items():
                params[name] = grad_sum / self.world_size
                # Reset cache
                self.parameter_cache[name] = None
            return params
        else:
            # Request from server
            return {}
            

class DistributedDataLoader:
    """Distributed data loader that shards data across processes."""
    
    def __init__(self, dataset: List[Any], batch_size: int, rank: int, world_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        
        # Shard dataset
        self.indices = list(range(rank, len(dataset), world_size))
        self.num_batches = len(self.indices) // batch_size
        
    def __iter__(self):
        """Iterate over sharded batches."""
        # Shuffle indices
        np.random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield batch
            
    def __len__(self):
        return self.num_batches
        

class DistributedTrainer:
    """Distributed trainer for MLX models."""
    
    def __init__(self, base_trainer, distributed_config: DistributedConfig):
        self.base_trainer = base_trainer
        self.config = distributed_config
        
        # Initialize backend
        if self.config.backend == "allreduce":
            self.backend = AllReduceBackend(self.config.world_size, self.config.rank)
        elif self.config.backend == "ring_allreduce":
            self.backend = RingAllReduceBackend(self.config.world_size, self.config.rank)
        elif self.config.backend == "parameter_server":
            is_server = self.config.rank == 0
            self.backend = ParameterServerBackend(self.config.world_size, self.config.rank, is_server)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
            
    def setup(self):
        """Setup distributed training."""
        # Initialize process group
        if hasattr(self.backend, 'init_process_group'):
            self.backend.init_process_group()
            
        # Load model on each process
        self.base_trainer.load_model()
        
        # Broadcast initial weights from rank 0
        if self.config.rank == 0:
            print(f"Broadcasting initial weights from rank 0")
            
        self._broadcast_parameters()
        
        # Setup distributed data loader
        train_dataset = self.base_trainer.train_dataset
        self.train_loader = DistributedDataLoader(
            train_dataset,
            self.base_trainer.config.batch_size,
            self.config.rank,
            self.config.world_size
        )
        
        print(f"Rank {self.config.rank}: Setup complete, handling {len(self.train_loader.indices)} samples")
        
    def _broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all processes."""
        params = dict(tree_flatten(self.base_trainer.model.parameters()))
        
        for name, param in params.items():
            if hasattr(self.backend, 'broadcast'):
                params[name] = self.backend.broadcast(param, src=0)
                
        # Update model parameters
        self.base_trainer.model.update(tree_unflatten(list(params.items())))
        
    def _all_reduce_gradients(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """All-reduce gradients across all processes."""
        flat_grads = dict(tree_flatten(gradients))
        
        for name, grad in flat_grads.items():
            if hasattr(self.backend, 'all_reduce'):
                flat_grads[name] = self.backend.all_reduce(grad, op="mean")
            elif hasattr(self.backend, 'ring_all_reduce'):
                flat_grads[name] = self.backend.ring_all_reduce(grad) / self.config.world_size
                
        return tree_unflatten(list(flat_grads.items()))
        
    def training_step(self, batch: Dict[str, Any]) -> float:
        """Distributed training step."""
        # Forward and backward pass
        def loss_fn(model):
            outputs = model(batch["input_ids"])
            logits = outputs.logits
            labels = batch["labels"]
            
            shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[..., 1:].reshape(-1)
            
            return nn.losses.cross_entropy(shift_logits, shift_labels)
            
        loss, grads = mx.value_and_grad(loss_fn)(self.base_trainer.model)
        
        # All-reduce gradients
        if self.config.backend == "parameter_server":
            # Push gradients to server
            self.backend.push_gradients(dict(tree_flatten(grads)))
            # Pull updated parameters
            new_params = self.backend.pull_parameters()
            if new_params:
                self.base_trainer.model.update(tree_unflatten(list(new_params.items())))
        else:
            # All-reduce gradients
            grads = self._all_reduce_gradients(grads)
            
            # Update weights
            self.base_trainer.optimizer.update(self.base_trainer.model, grads)
            
        mx.eval(loss)
        return loss.item()
        
    def train(self):
        """Distributed training loop."""
        print(f"Rank {self.config.rank}: Starting distributed training")
        
        self.base_trainer.model.train()
        total_loss = 0
        num_steps = 0
        
        for epoch in range(self.base_trainer.config.epochs):
            epoch_loss = 0
            epoch_steps = 0
            
            for batch_data in self.train_loader:
                batch = self.base_trainer.prepare_batch(batch_data)
                loss = self.training_step(batch)
                
                epoch_loss += loss
                epoch_steps += 1
                num_steps += 1
                
                if num_steps % self.base_trainer.config.logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    print(f"Rank {self.config.rank} | Epoch {epoch} | Step {num_steps} | Loss: {avg_loss:.4f}")
                    
            # Synchronize at end of epoch
            if hasattr(self.backend, 'barrier'):
                self.backend.barrier()
                
            # Save checkpoint from rank 0
            if self.config.rank == 0 and (epoch + 1) % self.base_trainer.config.save_epochs == 0:
                self.base_trainer.save_checkpoint(f"{self.base_trainer.config.output_dir}/epoch_{epoch+1}")
                
        # Final synchronization
        if hasattr(self.backend, 'barrier'):
            self.backend.barrier()
            
        return {
            "rank": self.config.rank,
            "final_loss": epoch_loss / epoch_steps if epoch_steps > 0 else 0,
            "total_steps": num_steps
        }


def launch_distributed_training(
    trainer_class,
    config,
    world_size: int,
    backend: str = "allreduce"
):
    """Launch distributed training across multiple processes."""
    
    def worker(rank: int, world_size: int):
        """Worker process for distributed training."""
        # Set environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        # Create distributed config
        dist_config = DistributedConfig(
            world_size=world_size,
            rank=rank,
            backend=backend
        )
        
        # Create base trainer
        base_trainer = trainer_class(config)
        
        # Create distributed trainer
        dist_trainer = DistributedTrainer(base_trainer, dist_config)
        
        # Setup and train
        dist_trainer.setup()
        results = dist_trainer.train()
        
        print(f"Rank {rank} completed: {results}")
        
    # Launch processes
    processes = []
    for rank in range(world_size):
        p = Process(target=worker, args=(rank, world_size))
        p.start()
        processes.append(p)
        
    # Wait for all processes
    for p in processes:
        p.join()
        
    print("Distributed training completed")