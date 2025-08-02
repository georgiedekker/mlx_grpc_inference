#!/usr/bin/env python3
"""
Test the system locally before distributed deployment.
"""

import sys
sys.path.append('.')

from src.core.config import ClusterConfig
from src.model.sharding import ModelShardingStrategy

def test_config():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    config = ClusterConfig.from_yaml("config/cluster_config.yaml")
    
    print(f"✓ Cluster name: {config.name}")
    print(f"✓ Coordinator: {config.coordinator_device_id}")
    print(f"✓ Total devices: {len(config.devices)}")
    
    for device in config.devices:
        print(f"  - {device.device_id}: {device.role.value} (rank {device.rank})")
    
    print(f"✓ Model: {config.model.name}")
    print(f"✓ Total layers: {config.model.total_layers}")
    
    return config


def test_sharding(config):
    """Test model sharding strategy."""
    print("\nTesting sharding strategy...")
    
    strategy = ModelShardingStrategy(config)
    
    if strategy.validate_coverage():
        print("✓ Layer coverage validated successfully")
    else:
        print("✗ Layer coverage validation failed")
        return False
    
    for device_id, shard in strategy.shards.items():
        print(f"  - {device_id}: layers {shard.layer_indices}")
    
    return True


def main():
    """Run local tests."""
    print("🧪 Running local tests for MLX distributed inference\n")
    
    try:
        # Test configuration
        config = test_config()
        
        # Test sharding
        if not test_sharding(config):
            print("\n❌ Tests failed")
            return 1
        
        print("\n✅ All local tests passed!")
        print("\nNext steps:")
        print("1. Copy this project to mini2 and master devices")
        print("2. Set up virtual environments on each device")
        print("3. Run: ./scripts/start_cluster.sh")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())