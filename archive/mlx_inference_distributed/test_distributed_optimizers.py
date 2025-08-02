"""
Test suite for distributed optimizers.

This module provides comprehensive tests for the distributed optimizer implementations,
including gradient synchronization, accumulation, and mixed precision training.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from typing import Dict, Any
import tempfile
import os

from .distributed_adamw import (
    DistributedAdamW,
    DistributedSGD,
    DistributedLion,
    OptimizerConfig,
    GradientSyncMode,
    GradientClipper,
    GradientNormalizer,
    create_distributed_optimizer
)


class MockCommunicator:
    """Mock communicator for testing without actual MPI."""
    
    def __init__(self, world_size: int = 1):
        self.world_size = world_size
    
    def allreduce(self, tensor: mx.array, op: str = "sum") -> mx.array:
        """Simulate allreduce by returning the same tensor."""
        return tensor
    
    def barrier(self) -> None:
        """Mock barrier operation."""
        pass


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = mx.relu(x)
        x = self.linear2(x)
        return x


def create_sample_gradients(model: nn.Module) -> Dict[str, mx.array]:
    """Create sample gradients for testing."""
    gradients = {}
    for name, param in model.parameters().items():
        gradients[name] = mx.random.normal(param.shape)
    return gradients


def test_gradient_clipper():
    """Test gradient clipping functionality."""
    gradients = {
        "layer1.weight": mx.array([[1.0, 2.0], [3.0, 4.0]]),
        "layer1.bias": mx.array([0.5, 1.5]),
        "layer2.weight": mx.array([[2.0, 1.0], [1.0, 2.0]]),
    }
    
    # Test global norm computation
    global_norm = GradientClipper.compute_global_norm(gradients)
    expected_norm = mx.sqrt(1.0 + 4.0 + 9.0 + 16.0 + 0.25 + 2.25 + 4.0 + 1.0 + 1.0 + 4.0)
    assert abs(float(global_norm) - float(expected_norm)) < 1e-6
    
    # Test gradient clipping
    clipped_gradients, actual_norm = GradientClipper.clip_by_global_norm(gradients, max_norm=1.0)
    clipped_norm = GradientClipper.compute_global_norm(clipped_gradients)
    assert float(clipped_norm) <= 1.0 + 1e-6
    assert abs(float(actual_norm) - float(expected_norm)) < 1e-6
    
    # Test gradient clipping by value
    value_clipped = GradientClipper.clip_by_value(gradients, clip_value=2.0)
    for grad in value_clipped.values():
        assert mx.all(grad >= -2.0) and mx.all(grad <= 2.0)


def test_gradient_normalizer():
    """Test gradient normalization functionality."""
    gradients = {
        "layer1.weight": mx.array([[1.0, 2.0], [3.0, 4.0]]),
        "layer1.bias": mx.array([0.5, 1.5]),
    }
    
    # Test global norm normalization
    normalized = GradientNormalizer.normalize_by_global_norm(gradients)
    normalized_norm = GradientClipper.compute_global_norm(normalized)
    assert abs(float(normalized_norm) - 1.0) < 1e-6
    
    # Test layer-wise normalization
    layer_normalized = GradientNormalizer.normalize_by_layer_norm(gradients)
    for name, grad in layer_normalized.items():
        layer_norm = mx.sqrt(mx.sum(grad * grad))
        assert abs(float(layer_norm) - 1.0) < 1e-6


def test_optimizer_config():
    """Test optimizer configuration."""
    config = OptimizerConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=4,
        use_mixed_precision=True
    )
    
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.01
    assert config.gradient_clip_norm == 1.0
    assert config.gradient_accumulation_steps == 4
    assert config.use_mixed_precision == True
    assert config.sync_mode == GradientSyncMode.ALLREDUCE


def test_distributed_adamw():
    """Test DistributedAdamW optimizer."""
    model = SimpleModel()
    config = OptimizerConfig(learning_rate=1e-3, gradient_accumulation_steps=2)
    communicator = MockCommunicator()
    
    optimizer = DistributedAdamW(
        config=config,
        communicator=communicator,
        rank=0,
        world_size=1
    )
    
    # Test basic functionality
    gradients = create_sample_gradients(model)
    
    # First update (should accumulate)
    stats1 = optimizer.update(model, gradients)
    assert stats1["should_update"] == False
    assert stats1["updated"] == False
    
    # Second update (should apply)
    stats2 = optimizer.update(model, gradients)
    assert stats2["should_update"] == True
    assert stats2["updated"] == True
    
    # Check optimizer state
    state = optimizer.state
    assert "loss_scaler" in state
    assert optimizer.accumulation_count == 0  # Should be reset after update


def test_distributed_sgd():
    """Test DistributedSGD optimizer."""
    model = SimpleModel()
    config = OptimizerConfig(learning_rate=1e-2, gradient_accumulation_steps=1)
    communicator = MockCommunicator()
    
    optimizer = DistributedSGD(
        learning_rate=1e-2,
        momentum=0.9,
        config=config,
        communicator=communicator,
        rank=0,
        world_size=1
    )
    
    gradients = create_sample_gradients(model)
    
    # Single update should apply immediately
    stats = optimizer.update(model, gradients)
    assert stats["should_update"] == True
    assert stats["updated"] == True
    
    # Check optimizer created base optimizer
    assert optimizer.base_optimizer is not None


def test_distributed_lion():
    """Test DistributedLion optimizer."""
    model = SimpleModel()
    config = OptimizerConfig(learning_rate=1e-4, gradient_accumulation_steps=1)
    communicator = MockCommunicator()
    
    optimizer = DistributedLion(
        learning_rate=1e-4,
        betas=(0.9, 0.99),
        config=config,
        communicator=communicator,
        rank=0,
        world_size=1
    )
    
    gradients = create_sample_gradients(model)
    
    # Test update
    stats = optimizer.update(model, gradients)
    assert stats["should_update"] == True
    assert stats["updated"] == True
    assert stats["step_count"] == 1
    
    # Check momentum state initialization
    assert optimizer.momentum_state is not None
    assert len(optimizer.momentum_state) > 0


def test_mixed_precision_training():
    """Test mixed precision training functionality."""
    model = SimpleModel()
    config = OptimizerConfig(
        learning_rate=1e-3,
        use_mixed_precision=True,
        mixed_precision_dtype="float16",
        loss_scale=1024.0,
        dynamic_loss_scaling=True
    )
    communicator = MockCommunicator()
    
    optimizer = DistributedAdamW(
        config=config,
        communicator=communicator,
        rank=0,
        world_size=1
    )
    
    # Test with normal gradients
    gradients = create_sample_gradients(model)
    stats = optimizer.update(model, gradients)
    assert stats["updated"] == True
    
    # Test with overflow gradients
    overflow_gradients = {}
    for name, grad in gradients.items():
        overflow_gradients[name] = grad * 1e10  # Create overflow
    
    stats_overflow = optimizer.update(model, overflow_gradients)
    assert stats_overflow["updated"] == False  # Should skip update due to overflow
    
    # Loss scaler should be reduced
    assert optimizer.loss_scaler < config.loss_scale


def test_gradient_accumulation():
    """Test gradient accumulation functionality."""
    model = SimpleModel()
    config = OptimizerConfig(
        learning_rate=1e-3,
        gradient_accumulation_steps=3
    )
    communicator = MockCommunicator()
    
    optimizer = DistributedAdamW(
        config=config,
        communicator=communicator,
        rank=0,
        world_size=1
    )
    
    gradients = create_sample_gradients(model)
    
    # First two updates should accumulate
    stats1 = optimizer.update(model, gradients)
    assert stats1["updated"] == False
    assert optimizer.accumulation_count == 1
    
    stats2 = optimizer.update(model, gradients)
    assert stats2["updated"] == False
    assert optimizer.accumulation_count == 2
    
    # Third update should apply
    stats3 = optimizer.update(model, gradients)
    assert stats3["updated"] == True
    assert optimizer.accumulation_count == 0  # Reset after update


def test_create_distributed_optimizer():
    """Test optimizer factory function."""
    config = OptimizerConfig(learning_rate=1e-3)
    communicator = MockCommunicator()
    
    # Test AdamW creation
    adamw = create_distributed_optimizer(
        "adamw", config, communicator, rank=0, world_size=1
    )
    assert isinstance(adamw, DistributedAdamW)
    
    # Test SGD creation
    sgd = create_distributed_optimizer(
        "sgd", config, communicator, rank=0, world_size=1
    )
    assert isinstance(sgd, DistributedSGD)
    
    # Test Lion creation
    lion = create_distributed_optimizer(
        "lion", config, communicator, rank=0, world_size=1
    )
    assert isinstance(lion, DistributedLion)
    
    # Test invalid optimizer
    with pytest.raises(ValueError):
        create_distributed_optimizer(
            "invalid", config, communicator, rank=0, world_size=1
        )


def test_optimizer_state_save_load():
    """Test optimizer state saving and loading."""
    model = SimpleModel()
    config = OptimizerConfig(learning_rate=1e-3)
    communicator = MockCommunicator()
    
    # Create and update optimizer
    optimizer = DistributedAdamW(
        config=config,
        communicator=communicator,
        rank=0,
        world_size=1
    )
    
    gradients = create_sample_gradients(model)
    optimizer.update(model, gradients)
    
    # Save state
    original_state = optimizer.state
    original_loss_scaler = optimizer.loss_scaler
    
    # Create new optimizer and load state  
    new_optimizer = DistributedAdamW(
        config=config,
        communicator=communicator,
        rank=0,
        world_size=1
    )
    
    new_optimizer.load_state(original_state)
    
    # Check state was loaded correctly
    assert new_optimizer.loss_scaler == original_loss_scaler


def test_performance_tracking():
    """Test performance tracking functionality."""
    model = SimpleModel()
    config = OptimizerConfig(learning_rate=1e-3)
    communicator = MockCommunicator()
    
    optimizer = DistributedAdamW(
        config=config,
        communicator=communicator,
        rank=0,
        world_size=1
    )
    
    # Perform multiple updates
    gradients = create_sample_gradients(model)
    for _ in range(5):
        optimizer.update(model, gradients)
    
    # Check stats
    stats = optimizer.get_stats()
    assert stats["total_updates"] == 5
    assert "avg_sync_time" in stats
    assert "avg_update_time" in stats
    assert stats["avg_update_time"] >= 0


if __name__ == "__main__":
    # Run tests
    print("Running distributed optimizer tests...")
    
    test_gradient_clipper()
    print("✓ Gradient clipper tests passed")
    
    test_gradient_normalizer()  
    print("✓ Gradient normalizer tests passed")
    
    test_optimizer_config()
    print("✓ Optimizer config tests passed")
    
    test_distributed_adamw()
    print("✓ DistributedAdamW tests passed")
    
    test_distributed_sgd()
    print("✓ DistributedSGD tests passed")
    
    test_distributed_lion()
    print("✓ DistributedLion tests passed")
    
    test_mixed_precision_training()
    print("✓ Mixed precision training tests passed")
    
    test_gradient_accumulation()
    print("✓ Gradient accumulation tests passed")
    
    test_create_distributed_optimizer()
    print("✓ Optimizer factory tests passed")
    
    test_optimizer_state_save_load()
    print("✓ State save/load tests passed")
    
    test_performance_tracking()
    print("✓ Performance tracking tests passed")
    
    print("\nAll tests passed! 🎉")