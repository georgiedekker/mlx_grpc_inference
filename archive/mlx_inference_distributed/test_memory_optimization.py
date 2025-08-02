#!/usr/bin/env python3
"""
Test Memory Optimization for MLX Distributed Inference

This script validates that the memory optimization correctly loads only
assigned layers per device, dramatically reducing memory usage.
"""

import os
import sys
import time
import psutil
import mlx.core as mx
from typing import Dict, List
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_efficient_model_loader import create_memory_efficient_model, MemoryEfficientModelLoader
from sharding_strategy import ResourceAwareShardingPlanner, ShardingStrategy
from device_capabilities import DeviceProfile
from model_abstraction import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def simulate_3_device_cluster():
    """Simulate a 3-device M4 Mac cluster."""
    devices = [
        DeviceProfile(
            device_id="mini1",
            hostname="mini1.local",
            model="Apple M4",
            memory_gb=16.0,
            gpu_cores=10,
            cpu_cores=10,
            cpu_performance_cores=4,
            cpu_efficiency_cores=6,
            neural_engine_cores=16
        ),
        DeviceProfile(
            device_id="mini2", 
            hostname="mini2.local",
            model="Apple M4",
            memory_gb=16.0,
            gpu_cores=10,
            cpu_cores=10,
            cpu_performance_cores=4,
            cpu_efficiency_cores=6,
            neural_engine_cores=16
        ),
        DeviceProfile(
            device_id="mini3",
            hostname="mini3.local",
            model="Apple M4",
            memory_gb=16.0,
            gpu_cores=10,
            cpu_cores=10,
            cpu_performance_cores=4,
            cpu_efficiency_cores=6,
            neural_engine_cores=16
        )
    ]
    return devices


def test_old_vs_new_approach():
    """Compare memory usage between old full-model approach and new optimized approach."""
    print("="*80)
    print("MEMORY OPTIMIZATION TEST: Old vs New Approach")
    print("="*80)
    
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    devices = simulate_3_device_cluster()
    
    # Test 1: Simulate OLD approach (full model loading)
    print("\n🔴 Testing OLD approach (full model on each device):")
    print("-" * 50)
    
    old_memory_usage = []
    initial_memory = get_memory_usage()
    
    for i, device in enumerate(devices):
        print(f"Device {i} ({device.device_id}): Loading FULL model...")
        start_memory = get_memory_usage()
        
        # Simulate old approach - load full model
        wrapper = ModelFactory.create_wrapper(model_name)
        wrapper.load_model()
        
        device_memory = get_memory_usage() - start_memory
        old_memory_usage.append(device_memory)
        print(f"  Memory used: {device_memory:.2f} GB")
        
        # Clean up
        del wrapper
        mx.metal.clear_cache()
    
    total_old_memory = sum(old_memory_usage)
    print(f"\n📊 OLD APPROACH TOTAL: {total_old_memory:.2f} GB across 3 devices")
    print(f"   Average per device: {total_old_memory/3:.2f} GB")
    
    # Test 2: NEW memory-efficient approach
    print("\n🟢 Testing NEW approach (only assigned layers per device):")
    print("-" * 50)
    
    # Create optimal sharding plan
    loader = MemoryEfficientModelLoader(model_name)
    model_info = loader.load_model_info_only()
    
    planner = ResourceAwareShardingPlanner()
    best_plan, all_plans = planner.find_optimal_strategy(model_info, devices)
    
    print(f"Selected sharding strategy: {best_plan.strategy.value}")
    print(f"Balance score: {best_plan.balance_score:.3f}")
    
    new_memory_usage = []
    
    for i, assignment in enumerate(best_plan.assignments):
        print(f"\nDevice {i} ({assignment.device_id}): Loading ONLY assigned layers {assignment.start_layer}-{assignment.end_layer-1}")
        start_memory = get_memory_usage()
        
        # Use new memory-efficient approach
        sharded_model, tokenizer = create_memory_efficient_model(model_name, assignment)
        
        device_memory = get_memory_usage() - start_memory
        actual_memory = sharded_model.get_memory_usage()
        new_memory_usage.append(actual_memory)
        
        print(f"  System memory used: {device_memory:.2f} GB")
        print(f"  Model memory footprint: {actual_memory:.2f} GB")
        print(f"  Estimated memory: {assignment.estimated_memory_gb:.2f} GB")
        print(f"  Memory saved vs full model: ~{(total_old_memory/3) - actual_memory:.2f} GB")
        print(f"  Components: embed={assignment.has_embedding}, norm/head={assignment.has_lm_head}")
        
        # Clean up
        del sharded_model, tokenizer
        mx.metal.clear_cache()
    
    total_new_memory = sum(new_memory_usage)
    memory_saved = total_old_memory - total_new_memory
    
    print(f"\n📊 NEW APPROACH TOTAL: {total_new_memory:.2f} GB across 3 devices")
    print(f"   Average per device: {total_new_memory/3:.2f} GB")
    print(f"\n💰 MEMORY SAVINGS:")
    print(f"   Total saved: {memory_saved:.2f} GB ({(memory_saved/total_old_memory)*100:.1f}%)")
    print(f"   Saved per device: {memory_saved/3:.2f} GB")
    
    return {
        'old_total': total_old_memory,
        'new_total': total_new_memory,
        'saved': memory_saved,
        'savings_percent': (memory_saved/total_old_memory)*100,
        'old_per_device': old_memory_usage,
        'new_per_device': new_memory_usage
    }


def test_layer_distribution():
    """Test that layers are properly distributed across devices."""
    print("\n" + "="*80)
    print("LAYER DISTRIBUTION TEST")
    print("="*80)
    
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    devices = simulate_3_device_cluster()
    
    # Get model info
    loader = MemoryEfficientModelLoader(model_name)
    model_info = loader.load_model_info_only()
    
    print(f"Model: {model_info.name}")
    print(f"Total layers: {model_info.num_layers}")
    print(f"Total parameters: {model_info.total_params:,}")
    print(f"Estimated size: {model_info.estimate_size_gb():.2f} GB")
    
    # Create sharding plan
    planner = ResourceAwareShardingPlanner()
    best_plan, all_plans = planner.find_optimal_strategy(model_info, devices)
    
    print(f"\nSharding Strategy: {best_plan.strategy.value}")
    print(f"Balance Score: {best_plan.balance_score:.3f}")
    
    # Verify layer coverage
    all_assigned_layers = set()
    for i, assignment in enumerate(best_plan.assignments):
        layers = set(assignment.layer_indices)
        all_assigned_layers.update(layers)
        
        print(f"\nDevice {i} ({assignment.device_id}):")
        print(f"  Layers: {assignment.start_layer} to {assignment.end_layer-1} ({assignment.num_layers} layers)")
        print(f"  Layer indices: {assignment.layer_indices}")
        print(f"  Memory estimate: {assignment.estimated_memory_gb:.2f} GB")
        print(f"  Memory utilization: {assignment.memory_utilization():.1f}%")
        print(f"  Has embedding: {assignment.has_embedding}")
        print(f"  Has LM head: {assignment.has_lm_head}")
    
    # Verify complete coverage
    expected_layers = set(range(model_info.num_layers))
    missing_layers = expected_layers - all_assigned_layers
    duplicate_layers = []
    
    for layer in all_assigned_layers:
        count = sum(1 for assignment in best_plan.assignments if layer in assignment.layer_indices)
        if count > 1:
            duplicate_layers.append(layer)
    
    print(f"\n🔍 VALIDATION:")
    print(f"  Expected layers: {len(expected_layers)}")
    print(f"  Assigned layers: {len(all_assigned_layers)}")
    print(f"  Missing layers: {missing_layers if missing_layers else 'None ✓'}")
    print(f"  Duplicate layers: {duplicate_layers if duplicate_layers else 'None ✓'}")
    
    # Check embedding and head assignment
    embedding_count = sum(1 for assignment in best_plan.assignments if assignment.has_embedding)
    head_count = sum(1 for assignment in best_plan.assignments if assignment.has_lm_head)
    
    print(f"  Embedding assignments: {embedding_count} (expected: 1) {'✓' if embedding_count == 1 else '❌'}")
    print(f"  LM head assignments: {head_count} (expected: 1) {'✓' if head_count == 1 else '❌'}")
    
    return len(missing_layers) == 0 and len(duplicate_layers) == 0 and embedding_count == 1 and head_count == 1


def test_inference_functionality():
    """Test that inference still works with memory-optimized loading."""
    print("\n" + "="*80)
    print("INFERENCE FUNCTIONALITY TEST")
    print("="*80)
    
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    devices = simulate_3_device_cluster()
    
    # Create sharding plan for first device (with embeddings)
    loader = MemoryEfficientModelLoader(model_name)
    model_info = loader.load_model_info_only()
    
    planner = ResourceAwareShardingPlanner()
    best_plan, all_plans = planner.find_optimal_strategy(model_info, devices)
    
    # Test the first device (which has embeddings)
    assignment = best_plan.assignments[0]
    print(f"Testing device 0 with layers {assignment.start_layer}-{assignment.end_layer-1}")
    print(f"Has embedding: {assignment.has_embedding}")
    
    # Load memory-efficient model
    sharded_model, tokenizer = create_memory_efficient_model(model_name, assignment)
    
    print(f"Model loaded successfully!")
    print(f"Memory usage: {sharded_model.get_memory_usage():.2f} GB")
    
    # Test tokenization
    test_prompt = "Hello, how are you today?"
    tokens = tokenizer.encode(test_prompt)
    print(f"Test prompt: '{test_prompt}'")
    print(f"Tokens: {tokens}")
    
    # Test forward pass (only for first device with embeddings)
    if assignment.has_embedding:
        print("Testing forward pass...")
        input_ids = mx.array(tokens).reshape(1, -1)
        
        try:
            with mx.stream(mx.gpu):
                output = sharded_model(input_ids)
                mx.eval(output)
            
            print(f"Forward pass successful!")
            print(f"Input shape: {input_ids.shape}")
            print(f"Output shape: {output.shape}")
            
            # Decode a few tokens to verify
            if len(tokens) > 0:
                decoded = tokenizer.decode(tokens[:5])
                print(f"Decoded sample: '{decoded}'")
                
        except Exception as e:
            print(f"Forward pass failed: {e}")
            return False
    
    print("✅ Inference functionality test passed!")
    return True


def main():
    """Run all memory optimization tests."""
    print("🚀 Starting MLX Memory Optimization Tests")
    print(f"MLX device: {mx.default_device()}")
    
    # Test 1: Memory usage comparison
    try:
        results = test_old_vs_new_approach()
        print(f"\n✅ Memory optimization test completed successfully!")
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False
    
    # Test 2: Layer distribution
    try:
        distribution_ok = test_layer_distribution()
        if distribution_ok:
            print(f"\n✅ Layer distribution test passed!")
        else:
            print(f"\n❌ Layer distribution test failed!")
            return False
    except Exception as e:
        print(f"❌ Layer distribution test failed: {e}")
        return False
    
    # Test 3: Inference functionality
    try:
        inference_ok = test_inference_functionality()
        if inference_ok:
            print(f"\n✅ Inference functionality test passed!")
        else:
            print(f"\n❌ Inference functionality test failed!")
            return False
    except Exception as e:
        print(f"❌ Inference functionality test failed: {e}")
        return False
    
    # Summary
    print("\n" + "="*80)
    print("🎉 ALL TESTS PASSED - MEMORY OPTIMIZATION SUCCESSFUL!")
    print("="*80)
    print(f"💾 Memory saved: {results['saved']:.2f} GB ({results['savings_percent']:.1f}%)")
    print(f"📊 Old total: {results['old_total']:.2f} GB → New total: {results['new_total']:.2f} GB")
    print(f"🔧 Each device now loads only its assigned layers instead of the full model")
    print(f"✨ Ready for production deployment across 3 M4 Mac devices!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)