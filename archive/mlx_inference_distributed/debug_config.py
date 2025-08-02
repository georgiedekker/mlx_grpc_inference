#!/usr/bin/env python3
"""Debug the configuration to see why it's detecting single device mode."""

import json
import requests

def debug_configuration():
    print("🔍 Debugging MLX Distributed Configuration")
    print("=" * 60)
    
    # Check config file
    with open('distributed_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"📁 Config file says:")
    print(f"  - model_parallel_size: {config['model_parallel_size']}")
    print(f"  - Number of devices: {len(config['device_list'])}")
    print(f"  - Device hostnames:")
    for device in config['device_list']:
        print(f"    * {device['device_id']}: {device['hostname']} (role: {device['role']})")
    
    # Check what the running system reports
    print(f"\n🌐 Running system reports:")
    try:
        gpu_info = requests.get("http://localhost:8100/distributed/gpu-info")
        if gpu_info.status_code == 200:
            data = gpu_info.json()
            cluster_info = data.get('cluster_info', {})
            print(f"  - total_devices: {cluster_info.get('total_devices')}")
            print(f"  - healthy_devices: {cluster_info.get('healthy_devices')}")
            print(f"  - world_size: {cluster_info.get('world_size')}")
            
            devices = data.get('devices', [])
            print(f"  - Active devices: {len(devices)}")
            for device in devices:
                print(f"    * {device['device_id']} (rank {device['rank']})")
        else:
            print(f"  - Could not get GPU info (status: {gpu_info.status_code})")
    except Exception as e:
        print(f"  - Error getting GPU info: {e}")
    
    print(f"\n🧐 Theory:")
    print(f"  The code has a check: if self.config.model_parallel_size == 1")
    print(f"  But the config file shows model_parallel_size = {config['model_parallel_size']}")
    print(f"  This suggests either:")
    print(f"    1. The config is being modified at runtime")
    print(f"    2. The world_size detection is overriding the config")
    print(f"    3. Device connectivity is failing and it's falling back")
    
    print(f"\n💡 The fallback code says:")
    print(f"  'Single device fallback - use full model generation'")
    print(f"  This means it's NOT using distributed inference at all!")
    print(f"  Instead, it's using MLX's built-in generate() function on device 0.")

if __name__ == "__main__":
    debug_configuration()