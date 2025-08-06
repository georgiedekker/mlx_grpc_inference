#!/usr/bin/env python3
"""
Test script for tensor parallelism implementation.
Validates AllReduce operations and model sharding.
"""
import asyncio
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
import logging
import sys
import time

# Add project root to path
sys.path.append('/Users/mini1/Movies/mlx_grpc_inference')

from src.tensor_parallel import (
    TensorParallelConfig, 
    AllReduceManager,
    TensorParallelAttention,
    TensorParallelMLP,
    shard_model_weights
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_allreduce():
    """Test AllReduce operation without actual network communication."""
    logger.info("=" * 60)
    logger.info("Testing AllReduce operations")
    logger.info("=" * 60)
    
    # Create mock AllReduce manager (single device for testing)
    all_reduce = AllReduceManager(device_id=0, world_size=1, worker_stubs=[])
    
    # Test tensor
    test_tensor = mx.random.normal((2, 10, 2048))
    logger.info(f"Test tensor shape: {test_tensor.shape}")
    logger.info(f"Test tensor mean: {test_tensor.mean():.4f}, std: {test_tensor.std():.4f}")
    
    # Since world_size=1, AllReduce should return the same tensor
    result = await all_reduce.all_reduce_sum(test_tensor, "test_session")
    
    assert mx.allclose(result, test_tensor), "AllReduce with world_size=1 should return same tensor"
    logger.info("✅ AllReduce test passed (single device)")
    
    return True


async def test_tensor_parallel_attention():
    """Test tensor parallel attention layer."""
    logger.info("=" * 60)
    logger.info("Testing Tensor Parallel Attention")
    logger.info("=" * 60)
    
    # Configuration
    config = TensorParallelConfig(
        device_id=0,
        world_size=2,  # Simulate 2 devices
        hidden_size=2048,
        num_attention_heads=16,
        intermediate_size=5632,
        head_dim=128
    )
    
    logger.info(f"Config: {config.heads_per_device} heads per device, {config.local_head_dim} local hidden dim")
    
    # Create mock AllReduce
    all_reduce = AllReduceManager(device_id=0, world_size=1, worker_stubs=[])  # Single device for testing
    
    # Create attention layer
    attention = TensorParallelAttention(config, all_reduce)
    
    # Test input
    batch_size, seq_len = 1, 10
    x = mx.random.normal((batch_size, seq_len, config.hidden_size))
    
    logger.info(f"Input shape: {x.shape}")
    
    # Forward pass
    output = await attention(x, session_id="test")
    
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, config.hidden_size), "Output shape mismatch"
    
    logger.info("✅ Tensor Parallel Attention test passed")
    return True


async def test_tensor_parallel_mlp():
    """Test tensor parallel MLP layer."""
    logger.info("=" * 60)
    logger.info("Testing Tensor Parallel MLP")
    logger.info("=" * 60)
    
    # Configuration
    config = TensorParallelConfig(
        device_id=0,
        world_size=2,
        hidden_size=2048,
        num_attention_heads=16,
        intermediate_size=5632,
        head_dim=128
    )
    
    logger.info(f"Config: local intermediate size = {config.local_intermediate_size}")
    
    # Create mock AllReduce
    all_reduce = AllReduceManager(device_id=0, world_size=1, worker_stubs=[])
    
    # Create MLP layer
    mlp = TensorParallelMLP(config, all_reduce)
    
    # Test input
    batch_size, seq_len = 1, 10
    x = mx.random.normal((batch_size, seq_len, config.hidden_size))
    
    logger.info(f"Input shape: {x.shape}")
    
    # Forward pass
    output = await mlp(x, session_id="test")
    
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, config.hidden_size), "Output shape mismatch"
    
    logger.info("✅ Tensor Parallel MLP test passed")
    return True


def test_model_sharding():
    """Test model weight sharding."""
    logger.info("=" * 60)
    logger.info("Testing Model Weight Sharding")
    logger.info("=" * 60)
    
    # Create a simple mock model structure
    class MockAttention:
        def __init__(self):
            self.q_proj = nn.Linear(2048, 2048, bias=False)
            self.k_proj = nn.Linear(2048, 2048, bias=False)
            self.v_proj = nn.Linear(2048, 2048, bias=False)
            self.o_proj = nn.Linear(2048, 2048, bias=False)
    
    class MockMLP:
        def __init__(self):
            self.gate_proj = nn.Linear(2048, 5632, bias=False)
            self.up_proj = nn.Linear(2048, 5632, bias=False)
            self.down_proj = nn.Linear(5632, 2048, bias=False)
    
    class MockLayer:
        def __init__(self):
            self.self_attn = MockAttention()
            self.mlp = MockMLP()
            self.input_layernorm = nn.RMSNorm(2048)
            self.post_attention_layernorm = nn.RMSNorm(2048)
    
    class MockModel:
        def __init__(self):
            self.model = type('obj', (object,), {
                'layers': [MockLayer() for _ in range(2)]  # 2 layers for testing
            })()
    
    model = MockModel()
    world_size = 2
    
    # Shard the model
    shards = shard_model_weights(model, world_size)
    
    # Verify sharding
    assert len(shards) == world_size, f"Expected {world_size} shards"
    
    for device_id, shard in shards.items():
        logger.info(f"Device {device_id}: {len(shard)} weight tensors")
        
        # Check a few key weights
        for layer_idx in range(2):
            # Check attention weight shapes
            q_key = f"layer.{layer_idx}.self_attn.q_proj.weight"
            if q_key in shard:
                weight = shard[q_key]
                expected_shape = (1024, 2048)  # Split output dimension
                assert weight.shape == expected_shape, f"Q projection shape mismatch: {weight.shape} != {expected_shape}"
            
            # Check MLP weight shapes
            gate_key = f"layer.{layer_idx}.mlp.gate_proj.weight"
            if gate_key in shard:
                weight = shard[gate_key]
                expected_shape = (2816, 2048)  # Split output dimension
                assert weight.shape == expected_shape, f"Gate projection shape mismatch: {weight.shape} != {expected_shape}"
    
    logger.info("✅ Model sharding test passed")
    return True


async def benchmark_tensor_parallel():
    """Benchmark tensor parallel operations."""
    logger.info("=" * 60)
    logger.info("Benchmarking Tensor Parallel Performance")
    logger.info("=" * 60)
    
    config = TensorParallelConfig(
        device_id=0,
        world_size=2,
        hidden_size=2048,
        num_attention_heads=16,
        intermediate_size=5632,
        head_dim=128
    )
    
    all_reduce = AllReduceManager(device_id=0, world_size=1, worker_stubs=[])
    
    # Create layers
    attention = TensorParallelAttention(config, all_reduce)
    mlp = TensorParallelMLP(config, all_reduce)
    
    # Test data
    batch_size, seq_len = 1, 100  # 100 tokens
    x = mx.random.normal((batch_size, seq_len, config.hidden_size))
    
    # Warmup
    for _ in range(3):
        _ = await attention(x, session_id="warmup")
        _ = await mlp(x, session_id="warmup")
    
    # Benchmark attention
    num_iterations = 10
    start = time.time()
    for i in range(num_iterations):
        _ = await attention(x, session_id=f"bench_{i}")
    attention_time = (time.time() - start) / num_iterations
    
    # Benchmark MLP
    start = time.time()
    for i in range(num_iterations):
        _ = await mlp(x, session_id=f"bench_{i}")
    mlp_time = (time.time() - start) / num_iterations
    
    logger.info(f"Attention forward pass: {attention_time*1000:.2f}ms")
    logger.info(f"MLP forward pass: {mlp_time*1000:.2f}ms")
    logger.info(f"Total per layer: {(attention_time + mlp_time)*1000:.2f}ms")
    logger.info(f"For 28 layers: {(attention_time + mlp_time)*28*1000:.2f}ms")
    
    # Calculate expected speedup
    pipeline_time = 133  # ms (from previous measurements)
    tensor_parallel_time = (attention_time + mlp_time) * 28 * 1000  # ms
    
    logger.info(f"\nExpected performance comparison:")
    logger.info(f"Pipeline parallelism: {pipeline_time:.1f}ms")
    logger.info(f"Tensor parallelism: {tensor_parallel_time:.1f}ms")
    
    if tensor_parallel_time < pipeline_time:
        speedup = (pipeline_time / tensor_parallel_time - 1) * 100
        logger.info(f"Expected speedup: {speedup:.1f}%")
    else:
        slowdown = (tensor_parallel_time / pipeline_time - 1) * 100
        logger.info(f"Expected slowdown: {slowdown:.1f}% (due to no actual parallelism in test)")
    
    return True


async def main():
    """Run all tests."""
    logger.info("Starting Tensor Parallelism Tests")
    logger.info("=" * 60)
    
    tests = [
        ("AllReduce", test_allreduce),
        ("Tensor Parallel Attention", test_tensor_parallel_attention),
        ("Tensor Parallel MLP", test_tensor_parallel_mlp),
        ("Model Sharding", lambda: asyncio.create_task(asyncio.to_thread(test_model_sharding))),
        ("Performance Benchmark", benchmark_tensor_parallel),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                if asyncio.iscoroutine(result):
                    result = await result
            
            if result:
                passed += 1
                logger.info(f"✅ {test_name} PASSED\n")
            else:
                failed += 1
                logger.info(f"❌ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with error: {e}\n")
    
    logger.info("=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"⚠️  {failed} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)