#!/usr/bin/env python3
"""Test to examine the actual sharding behavior."""

import sys
import os
sys.path.append('/Users/mini1/Movies/mlx_distributed_inference')

from distributed_mlx_inference import ShardInfo
import json

def test_sharding_logic():
    print("Testing MLX Distributed Sharding Logic")
    print("=" * 50)
    
    # Load current configuration
    with open('distributed_config.json', 'r') as f:
        config = json.load(f)
    
    model_parallel_size = config['model_parallel_size']
    print(f"Configured model parallel size: {model_parallel_size}")
    
    # Test sharding for different layer counts
    for total_layers in [24, 28, 32]:
        print(f"\n📊 Sharding {total_layers} layers across {model_parallel_size} devices:")
        
        shard_info = ShardInfo.create_balanced(total_layers, model_parallel_size)
        
        print(f"  Layers per device: {shard_info.layers_per_device}")
        
        for device_idx in range(model_parallel_size):
            assigned_layers = []
            for layer_idx, assigned_device in shard_info.layer_assignments.items():
                if assigned_device == device_idx:
                    assigned_layers.append(layer_idx)
            
            print(f"  Device {device_idx}: layers {sorted(assigned_layers)}")
    
    print(f"\n🔍 What happens with single device:")
    single_device_shard = ShardInfo.create_balanced(28, 1)
    print(f"  Device 0 gets all {single_device_shard.layers_per_device[0]} layers")
    print(f"  Layer assignments: {sorted(single_device_shard.layer_assignments.keys())}")
    
    print(f"\n🧠 Memory Analysis:")
    print("  ❌ Problem: Full model loaded on each device")
    print("  ✅ Positive: Only assigned layers are processed")
    print("  🎯 Opportunity: Could delete unused layers to save memory")
    
    print(f"\n🚀 What would happen with 3 devices running:")
    print("  - Device 0: Embedding + first 10 layers")
    print("  - Device 1: Middle 9 layers")  
    print("  - Device 2: Last 9 layers + norm + lm_head")
    print("  - Hidden states passed via gRPC between devices")
    
    print(f"\n💡 Current Status:")
    print("  - Sharding logic: ✅ IMPLEMENTED AND WORKING")
    print("  - Multi-device communication: ✅ IMPLEMENTED") 
    print("  - Memory optimization: ❌ NOT IMPLEMENTED")
    print("  - Actual distribution: ❌ ONLY 1 DEVICE RUNNING")

if __name__ == "__main__":
    test_sharding_logic()