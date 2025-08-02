"""
Distributed Optimizers for MLX

This module provides distributed versions of popular optimizers that can synchronize
gradients across multiple ranks in a distributed training setup. All optimizers
support gradient accumulation, mixed precision training, and efficient AllReduce
operations.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum

# Import communication infrastructure
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

try:
    from distributed_comm import (
        DistributedCommunicator, 
        CommunicationType
    )
except ImportError:
    # Fallback for testing
    class DistributedCommunicator:
        def allreduce(self, tensor, op="sum"):
            return tensor
        def barrier(self):
            pass

logger = logging.getLogger(__name__)


class GradientSyncMode(Enum):
    """Gradient synchronization modes."""
    ALLREDUCE = "allreduce"
    REDUCE_SCATTER = "reduce_scatter"
    PARAMETER_SERVER = "parameter_server"
    RING_ALLREDUCE = "ring_allreduce"


@dataclass
class OptimizerConfig:
    """Configuration for distributed optimizers."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    sync_mode: GradientSyncMode = GradientSyncMode.ALLREDUCE
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"
    loss_scale: float = 1024.0
    dynamic_loss_scaling: bool = True
    loss_scale_window: int = 2000
    min_loss_scale: float = 1.0
    max_loss_scale: float = 65536.0


class GradientClipper:
    """Utility class for gradient clipping operations."""
    
    @staticmethod
    def clip_by_global_norm(gradients: Dict[str, mx.array], max_norm: float) -> Tuple[Dict[str, mx.array], float]:
        """
        Clip gradients by global norm.
        
        Args:
            gradients: Dictionary of parameter gradients
            max_norm: Maximum allowed global norm
            
        Returns:
            Tuple of (clipped_gradients, global_norm)
        """
        if max_norm <= 0:
            total_norm = GradientClipper.compute_global_norm(gradients)
            return gradients, float(total_norm)
        
        # Calculate global gradient norm
        total_norm_squared = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm_squared += mx.sum(grad * grad)
        
        total_norm = mx.sqrt(total_norm_squared)
        
        # Apply clipping
        clip_coeff = max_norm / (total_norm + 1e-8)
        clip_coeff = mx.minimum(clip_coeff, 1.0)
        
        clipped_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                clipped_gradients[name] = grad * clip_coeff
            else:
                clipped_gradients[name] = grad
        
        return clipped_gradients, float(total_norm)
    
    @staticmethod
    def clip_by_value(gradients: Dict[str, mx.array], clip_value: float) -> Dict[str, mx.array]:
        """
        Clip gradients by value.
        
        Args:
            gradients: Dictionary of parameter gradients
            clip_value: Maximum absolute value for gradients
            
        Returns:
            Dictionary of clipped gradients
        """
        clipped_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                clipped_gradients[name] = mx.clip(grad, -clip_value, clip_value)
            else:
                clipped_gradients[name] = grad
        
        return clipped_gradients
    
    @staticmethod
    def compute_global_norm(gradients: Dict[str, mx.array]) -> mx.array:
        """Compute the global norm of gradients."""
        total_norm_squared = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm_squared += mx.sum(grad * grad)
        
        return mx.sqrt(total_norm_squared)


class GradientNormalizer:
    """Utility class for gradient normalization operations."""
    
    @staticmethod
    def normalize_by_global_norm(gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Normalize gradients by their global norm."""
        global_norm = GradientClipper.compute_global_norm(gradients)
        
        normalized_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                normalized_gradients[name] = grad / (global_norm + 1e-8)
            else:
                normalized_gradients[name] = grad
        
        return normalized_gradients
    
    @staticmethod
    def normalize_by_layer_norm(gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Normalize gradients by their layer-wise norm."""
        normalized_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                layer_norm = mx.sqrt(mx.sum(grad * grad))
                normalized_gradients[name] = grad / (layer_norm + 1e-8)
            else:
                normalized_gradients[name] = grad
        
        return normalized_gradients


class DistributedOptimizer(ABC):
    """
    Abstract base class for distributed optimizers.
    
    Provides common functionality for gradient synchronization, accumulation,
    and mixed precision training across distributed ranks.
    """
    
    def __init__(
        self,
        config: OptimizerConfig,
        communicator: Optional[DistributedCommunicator] = None,
        rank: int = 0,
        world_size: int = 1
    ):
        self.config = config
        self.communicator = communicator
        self.rank = rank
        self.world_size = world_size
        
        # Gradient accumulation state
        self.accumulated_gradients: Optional[Dict[str, mx.array]] = None
        self.accumulation_count = 0
        
        # Mixed precision state
        self.loss_scaler = config.loss_scale if config.use_mixed_precision else 1.0
        self.loss_scale_window_left = config.loss_scale_window
        self.loss_scale_updates = 0
        
        # Performance tracking
        self.sync_times = []
        self.update_times = []
        
        logger.info(f"Initialized {self.__class__.__name__} for rank {rank}/{world_size}")
    
    @abstractmethod
    def create_base_optimizer(self) -> optim.Optimizer:
        """Create the underlying MLX optimizer."""
        pass
    
    def sync_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Synchronize gradients across all ranks.
        
        Args:
            gradients: Dictionary of parameter gradients
            
        Returns:
            Synchronized gradients
        """
        if self.world_size == 1 or self.communicator is None:
            return gradients
        
        start_time = time.time()
        
        if self.config.sync_mode == GradientSyncMode.ALLREDUCE:
            synced_gradients = {}
            for name, grad in gradients.items():
                if grad is not None:
                    # Perform allreduce and average
                    synced_grad = self.communicator.allreduce(grad, op="sum")
                    synced_gradients[name] = synced_grad / self.world_size
                else:
                    synced_gradients[name] = grad
        
        elif self.config.sync_mode == GradientSyncMode.REDUCE_SCATTER:
            # Implement reduce-scatter for memory efficiency
            synced_gradients = self._reduce_scatter_gradients(gradients)
        
        else:
            raise NotImplementedError(f"Sync mode {self.config.sync_mode} not implemented")
        
        sync_time = time.time() - start_time
        self.sync_times.append(sync_time)
        
        return synced_gradients
    
    def _reduce_scatter_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Implement reduce-scatter for memory-efficient gradient synchronization.
        Each rank gets a subset of the averaged gradients.
        """
        # This is a simplified implementation
        # In practice, you'd partition gradients across ranks and only reduce relevant parts
        return self.sync_gradients(gradients)  # Fallback to allreduce for now
    
    def accumulate_gradients(self, gradients: Dict[str, mx.array]) -> bool:
        """
        Accumulate gradients over multiple steps.
        
        Args:
            gradients: Dictionary of parameter gradients
            
        Returns:
            True if gradients should be applied, False otherwise
        """
        if self.accumulated_gradients is None:
            self.accumulated_gradients = {}
            for name, grad in gradients.items():
                if grad is not None:
                    self.accumulated_gradients[name] = mx.zeros_like(grad)
                else:
                    self.accumulated_gradients[name] = None
        
        # Add to accumulated gradients
        for name, grad in gradients.items():
            if grad is not None and self.accumulated_gradients[name] is not None:
                self.accumulated_gradients[name] += grad
        
        self.accumulation_count += 1
        
        # Check if we should apply gradients
        if self.accumulation_count >= self.config.gradient_accumulation_steps:
            # Average accumulated gradients
            for name in self.accumulated_gradients:
                if self.accumulated_gradients[name] is not None:
                    self.accumulated_gradients[name] /= self.config.gradient_accumulation_steps
            
            return True
        
        return False
    
    def unscale_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Unscale gradients for mixed precision training."""
        if not self.config.use_mixed_precision or self.loss_scaler == 1.0:
            return gradients
        
        unscaled_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                unscaled_gradients[name] = grad / self.loss_scaler
            else:
                unscaled_gradients[name] = grad
        
        return unscaled_gradients
    
    def check_overflow(self, gradients: Dict[str, mx.array]) -> bool:
        """Check if gradients contain inf or nan values."""
        for grad in gradients.values():
            if grad is not None:
                if mx.any(mx.isinf(grad)) or mx.any(mx.isnan(grad)):
                    return True
        return False
    
    def update_loss_scale(self, overflow: bool) -> None:
        """Update loss scale for dynamic loss scaling."""
        if not self.config.dynamic_loss_scaling:
            return
        
        if overflow:
            # Reduce loss scale
            self.loss_scaler = max(self.loss_scaler / 2.0, self.config.min_loss_scale)
            self.loss_scale_window_left = self.config.loss_scale_window
            logger.warning(f"Gradient overflow detected, reducing loss scale to {self.loss_scaler}")
        else:
            self.loss_scale_window_left -= 1
            if self.loss_scale_window_left <= 0:
                # Increase loss scale
                self.loss_scaler = min(self.loss_scaler * 2.0, self.config.max_loss_scale)
                self.loss_scale_window_left = self.config.loss_scale_window
                logger.info(f"Increasing loss scale to {self.loss_scaler}")
    
    def process_gradients(self, gradients: Dict[str, mx.array]) -> Tuple[Dict[str, mx.array], bool]:
        """
        Process gradients through the full pipeline.
        
        Args:
            gradients: Raw gradients from backward pass
            
        Returns:
            Tuple of (processed_gradients, should_update)
        """
        # Unscale gradients if using mixed precision
        gradients = self.unscale_gradients(gradients)
        
        # Check for overflow
        overflow = self.check_overflow(gradients)
        self.update_loss_scale(overflow)
        
        if overflow:
            # Skip update if overflow detected
            self._reset_accumulation()
            return gradients, False
        
        # Accumulate gradients
        should_update = self.accumulate_gradients(gradients)
        
        if should_update:
            # Synchronize gradients across ranks
            synced_gradients = self.sync_gradients(self.accumulated_gradients)
            
            # Apply gradient clipping
            clipped_gradients, grad_norm = GradientClipper.clip_by_global_norm(
                synced_gradients, self.config.gradient_clip_norm
            )
            
            return clipped_gradients, True
        
        return gradients, False
    
    def _reset_accumulation(self) -> None:
        """Reset gradient accumulation state."""
        self.accumulated_gradients = None
        self.accumulation_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "loss_scaler": self.loss_scaler,
            "accumulation_count": self.accumulation_count,
            "avg_sync_time": np.mean(self.sync_times) if self.sync_times else 0.0,
            "avg_update_time": np.mean(self.update_times) if self.update_times else 0.0,
            "total_updates": len(self.update_times),
        }


class DistributedAdamW(DistributedOptimizer):
    """
    Distributed version of AdamW optimizer.
    
    Extends MLX's AdamW optimizer with distributed gradient synchronization,
    gradient accumulation, and mixed precision training support.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        config: Optional[OptimizerConfig] = None,
        communicator: Optional[DistributedCommunicator] = None,
        rank: int = 0,
        world_size: int = 1
    ):
        if config is None:
            config = OptimizerConfig(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        else:
            config.learning_rate = learning_rate
            config.weight_decay = weight_decay
        
        super().__init__(config, communicator, rank, world_size)
        
        self.betas = betas
        self.eps = eps
        self.base_optimizer: Optional[optim.AdamW] = None
    
    def create_base_optimizer(self) -> optim.AdamW:
        """Create the underlying MLX AdamW optimizer."""
        self.base_optimizer = optim.AdamW(
            learning_rate=self.config.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.config.weight_decay
        )
        return self.base_optimizer
    
    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> Dict[str, Any]:
        """
        Update model parameters with distributed gradient synchronization.
        
        Args:
            model: The model to update
            gradients: Dictionary of parameter gradients
            
        Returns:
            Dictionary of update statistics
        """
        start_time = time.time()
        
        # Process gradients (accumulate, sync, clip)
        processed_gradients, should_update = self.process_gradients(gradients)
        
        stats = {
            "should_update": should_update,
            "gradient_norm": float(GradientClipper.compute_global_norm(gradients)),
        }
        
        if should_update:
            # Create base optimizer if not exists
            if self.base_optimizer is None:
                self.create_base_optimizer()
            
            # Apply update using base optimizer
            self.base_optimizer.update(model, processed_gradients)
            
            # Reset accumulation
            self._reset_accumulation()
            
            stats["updated"] = True
        else:
            stats["updated"] = False
        
        update_time = time.time() - start_time
        self.update_times.append(update_time)
        
        stats.update(self.get_stats())
        stats["update_time"] = update_time
        
        return stats
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get optimizer state."""
        base_state = self.base_optimizer.state if self.base_optimizer else {}
        return {
            **base_state,
            "loss_scaler": self.loss_scaler,
            "accumulation_count": self.accumulation_count,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load optimizer state."""
        if self.base_optimizer is None:
            self.create_base_optimizer()
        
        # Extract distributed optimizer state
        self.loss_scaler = state.get("loss_scaler", self.config.loss_scale)
        self.accumulation_count = state.get("accumulation_count", 0)
        
        # Load base optimizer state
        base_state = {k: v for k, v in state.items() 
                     if k not in ["loss_scaler", "accumulation_count"]}
        if base_state and hasattr(self.base_optimizer, 'load_state'):
            self.base_optimizer.load_state(base_state)


class DistributedSGD(DistributedOptimizer):
    """
    Distributed version of SGD optimizer with momentum.
    
    Provides distributed gradient synchronization for SGD with momentum,
    including gradient accumulation and mixed precision training support.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        config: Optional[OptimizerConfig] = None,
        communicator: Optional[DistributedCommunicator] = None,
        rank: int = 0,
        world_size: int = 1
    ):
        if config is None:
            config = OptimizerConfig(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        else:
            config.learning_rate = learning_rate
            config.weight_decay = weight_decay
        
        super().__init__(config, communicator, rank, world_size)
        
        self.momentum = momentum
        self.nesterov = nesterov
        self.base_optimizer: Optional[optim.SGD] = None
    
    def create_base_optimizer(self) -> optim.SGD:
        """Create the underlying MLX SGD optimizer."""
        self.base_optimizer = optim.SGD(
            learning_rate=self.config.learning_rate,
            momentum=self.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=self.nesterov
        )
        return self.base_optimizer
    
    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> Dict[str, Any]:
        """
        Update model parameters with distributed gradient synchronization.
        
        Args:
            model: The model to update
            gradients: Dictionary of parameter gradients
            
        Returns:
            Dictionary of update statistics
        """
        start_time = time.time()
        
        # Process gradients (accumulate, sync, clip)
        processed_gradients, should_update = self.process_gradients(gradients)
        
        stats = {
            "should_update": should_update,
            "gradient_norm": float(GradientClipper.compute_global_norm(gradients)),
        }
        
        if should_update:
            # Create base optimizer if not exists
            if self.base_optimizer is None:
                self.create_base_optimizer()
            
            # Apply update using base optimizer
            self.base_optimizer.update(model, processed_gradients)
            
            # Reset accumulation
            self._reset_accumulation()
            
            stats["updated"] = True
        else:
            stats["updated"] = False
        
        update_time = time.time() - start_time
        self.update_times.append(update_time)
        
        stats.update(self.get_stats())
        stats["update_time"] = update_time
        
        return stats
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get optimizer state."""
        base_state = self.base_optimizer.state if self.base_optimizer else {}
        return {
            **base_state,
            "loss_scaler": self.loss_scaler,
            "accumulation_count": self.accumulation_count,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load optimizer state."""
        if self.base_optimizer is None:
            self.create_base_optimizer()
        
        # Extract distributed optimizer state
        self.loss_scaler = state.get("loss_scaler", self.config.loss_scale)
        self.accumulation_count = state.get("accumulation_count", 0)
        
        # Load base optimizer state
        base_state = {k: v for k, v in state.items() 
                     if k not in ["loss_scaler", "accumulation_count"]}
        if base_state and hasattr(self.base_optimizer, 'load_state'):
            self.base_optimizer.load_state(base_state)


class DistributedLion(DistributedOptimizer):
    """
    Distributed version of Lion optimizer.
    
    Lion (EvoLved Sign Momentum) is a memory-efficient optimizer that only
    tracks momentum and uses the sign of gradients for updates.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
        config: Optional[OptimizerConfig] = None,
        communicator: Optional[DistributedCommunicator] = None,
        rank: int = 0,
        world_size: int = 1
    ):
        if config is None:
            config = OptimizerConfig(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        else:
            config.learning_rate = learning_rate
            config.weight_decay = weight_decay
        
        super().__init__(config, communicator, rank, world_size)
        
        self.betas = betas
        self.momentum_state: Optional[Dict[str, mx.array]] = None
        self.step_count = 0
    
    def create_base_optimizer(self) -> 'DistributedLion':
        """Lion doesn't have an MLX base optimizer, so we implement it ourselves."""
        return self
    
    def _initialize_momentum(self, gradients: Dict[str, mx.array]) -> None:
        """Initialize momentum state."""
        if self.momentum_state is None:
            self.momentum_state = {}
            for name, grad in gradients.items():
                if grad is not None:
                    self.momentum_state[name] = mx.zeros_like(grad)
    
    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> Dict[str, Any]:
        """
        Update model parameters using Lion algorithm with distributed synchronization.
        
        Args:
            model: The model to update
            gradients: Dictionary of parameter gradients
            
        Returns:
            Dictionary of update statistics
        """
        start_time = time.time()
        
        # Process gradients (accumulate, sync, clip)
        processed_gradients, should_update = self.process_gradients(gradients)
        
        stats = {
            "should_update": should_update,
            "gradient_norm": float(GradientClipper.compute_global_norm(gradients)),
        }
        
        if should_update:
            self._initialize_momentum(processed_gradients)
            self.step_count += 1
            
            # Get model parameters
            parameters = model.parameters()
            
            # Lion update rule
            beta1, beta2 = self.betas
            
            for name, param in parameters.items():
                if name in processed_gradients and processed_gradients[name] is not None:
                    grad = processed_gradients[name]
                    momentum = self.momentum_state[name]
                    
                    # Update rule: param = param - lr * sign(beta1 * momentum + (1 - beta1) * grad)
                    update_direction = mx.sign(beta1 * momentum + (1.0 - beta1) * grad)
                    
                    # Apply weight decay
                    if self.config.weight_decay > 0:
                        param_update = self.config.learning_rate * (update_direction + self.config.weight_decay * param)
                    else:
                        param_update = self.config.learning_rate * update_direction
                    
                    # Update parameter
                    param -= param_update
                    
                    # Update momentum: momentum = beta2 * momentum + (1 - beta2) * grad
                    self.momentum_state[name] = beta2 * momentum + (1.0 - beta2) * grad
            
            # Reset accumulation
            self._reset_accumulation()
            
            stats["updated"] = True
        else:
            stats["updated"] = False
        
        update_time = time.time() - start_time
        self.update_times.append(update_time)
        
        stats.update(self.get_stats())
        stats["update_time"] = update_time
        stats["step_count"] = self.step_count
        
        return stats
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            "momentum_state": self.momentum_state,
            "step_count": self.step_count,
            "loss_scaler": self.loss_scaler,
            "accumulation_count": self.accumulation_count,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load optimizer state."""
        self.momentum_state = state.get("momentum_state", None)
        self.step_count = state.get("step_count", 0)
        self.loss_scaler = state.get("loss_scaler", self.config.loss_scale)
        self.accumulation_count = state.get("accumulation_count", 0)


# Factory function for creating distributed optimizers
def create_distributed_optimizer(
    optimizer_name: str,
    config: OptimizerConfig,
    communicator: Optional[DistributedCommunicator] = None,
    rank: int = 0,
    world_size: int = 1,
    **optimizer_kwargs
) -> DistributedOptimizer:
    """
    Factory function to create distributed optimizers.
    
    Args:
        optimizer_name: Name of the optimizer ('adamw', 'sgd', 'lion')
        config: Optimizer configuration
        communicator: Distributed communicator
        rank: Current rank
        world_size: Total number of ranks
        **optimizer_kwargs: Additional optimizer-specific arguments
        
    Returns:
        Distributed optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adamw':
        return DistributedAdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            config=config,
            communicator=communicator,
            rank=rank,
            world_size=world_size,
            **optimizer_kwargs
        )
    elif optimizer_name == 'sgd':
        return DistributedSGD(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            config=config,
            communicator=communicator,
            rank=rank,
            world_size=world_size,
            **optimizer_kwargs
        )
    elif optimizer_name == 'lion':
        return DistributedLion(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            config=config,
            communicator=communicator,
            rank=rank,
            world_size=world_size,
            **optimizer_kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                        f"Supported optimizers: adamw, sgd, lion")


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = OptimizerConfig(
        learning_rate=1e-4,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=4,
        use_mixed_precision=True,
        mixed_precision_dtype="float16"
    )
    
    # Create distributed optimizer
    optimizer = create_distributed_optimizer(
        optimizer_name="adamw",
        config=config,
        rank=0,
        world_size=1
    )
    
    print(f"Created {optimizer.__class__.__name__} with config: {config}")
    print(f"Optimizer stats: {optimizer.get_stats()}")