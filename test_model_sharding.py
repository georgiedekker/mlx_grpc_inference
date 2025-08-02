#!/usr/bin/env python3
"""
Standalone test for model sharding without complex imports.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_sharding_strategy():
    """Test model sharding strategy."""
    print("🧠 Testing model sharding strategy...")
    
    try:
        # Import configuration
        sys.path.insert(0, str(Path(__file__).parent / "src" / "core"))
        sys.path.insert(0, str(Path(__file__).parent / "src" / "model"))
        
        from config import ClusterConfig
        from sharding import ModelShardingStrategy
        
        # Load config
        config = ClusterConfig.from_yaml("config/cluster_config.yaml")
        print(f"✅ Loaded config for {len(config.devices)} devices")
        
        # Create strategy
        strategy = ModelShardingStrategy(config)
        print(f"✅ Created sharding strategy")
        
        # Test coverage
        if strategy.validate_coverage():
            print("✅ Layer coverage validation passed")
        else:
            print("❌ Layer coverage validation failed")
            return False
        
        # Test device assignments
        for device in config.devices:
            shard = strategy.get_device_shard(device.device_id)
            if shard:
                print(f"   - {device.device_id}: layers {shard.layer_indices}")
            else:
                print(f"   - {device.device_id}: no layers assigned")
        
        # Test layer lookup
        layer_5_device = strategy.get_layer_device(5)
        print(f"   - Layer 5 assigned to: {layer_5_device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sharding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sharding_strategy()
    sys.exit(0 if success else 1)