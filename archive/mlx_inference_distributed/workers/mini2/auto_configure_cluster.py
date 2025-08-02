#!/usr/bin/env python3
"""
Automatically configure the distributed cluster by detecting hardware on all devices.
"""

import json
import subprocess
import os
from hardware_detector import HardwareDetector


def detect_local_hardware():
    """Detect hardware on local device."""
    detector = HardwareDetector()
    return detector.generate_device_config()


def detect_remote_hardware(hostname, username=None):
    """Detect hardware on remote device via SSH."""
    ssh_prefix = f"ssh {username}@{hostname}" if username else f"ssh {hostname}"
    
    # Run hardware detector remotely
    cmd = f'{ssh_prefix} "cd ~/Movies/mlx_distributed && source .venv/bin/activate && python hardware_detector.py"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Failed to detect hardware on {hostname}: {result.stderr}")
        return None
    
    # Parse the output to extract the JSON configuration
    output = result.stdout
    if "Generated Configuration:" in output:
        json_start = output.find("{", output.index("Generated Configuration:"))
        json_end = output.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = output[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from {hostname}: {e}")
                return None
    
    return None


def update_cluster_configuration():
    """Update the distributed configuration with detected hardware."""
    
    # Define cluster devices
    devices = [
        {"device_id": "mini1", "hostname": "mini1.local", "rank": 0, "role": "master", "port": 8100, "username": None},
        {"device_id": "mini2", "hostname": "mini2.local", "rank": 1, "role": "worker", "port": 8001, "username": None},
        {"device_id": "master", "hostname": "master.local", "rank": 2, "role": "worker", "port": 8002, "username": "georgedekker"}
    ]
    
    # Detect hardware for each device
    device_configs = []
    for device in devices:
        print(f"\n🔍 Detecting hardware for {device['device_id']} ({device['hostname']})...")
        
        if device['device_id'] == 'mini1':
            # Local device
            hw_config = detect_local_hardware()
        else:
            # Remote device
            hw_config = detect_remote_hardware(device['hostname'], device['username'])
        
        if hw_config:
            # Merge device info with hardware config
            device_config = {
                "device_id": device['device_id'],
                "hostname": device['hostname'],
                "port": device['port'],
                "role": device['role'],
                "device_index": device['rank'],
                "capabilities": {
                    "device_id": device['device_id'],
                    "hostname": device['hostname'],
                    "device_type": hw_config['device_type'],
                    "model": hw_config['model'],
                    "device_name": hw_config['device_name'],
                    "memory_gb": hw_config['memory_gb'],
                    "gpu_cores": hw_config['gpu_cores'],
                    "cpu_cores": hw_config['cpu_cores'],
                    "cpu_performance_cores": hw_config['cpu_performance_cores'],
                    "cpu_efficiency_cores": hw_config['cpu_efficiency_cores'],
                    "neural_engine_cores": hw_config['neural_engine_cores'],
                    "mlx_metal_available": hw_config['mlx_metal_available'],
                    "max_recommended_model_size_gb": hw_config['max_recommended_model_size_gb'],
                    "is_laptop": hw_config['is_laptop']
                }
            }
            device_configs.append(device_config)
            print(f"✅ Detected: {hw_config['device_name']} with {hw_config['memory_gb']}GB RAM, {hw_config['gpu_cores']} GPU cores")
        else:
            print(f"❌ Failed to detect hardware for {device['device_id']}")
    
    # Create updated configuration
    config = {
        "network_type": "lan",
        "master_hostname": "mini1.local",
        "master_port": 8100,
        "mpi_hosts": ["mini1.local", "mini2.local", "master.local"],
        "mpi_slots_per_host": 1,
        "model_name": "mlx-community/Qwen3-1.7B-8bit",
        "model_parallel_size": len(device_configs),
        "pipeline_parallel_size": 1,
        "communication_backend": "grpc",
        "heartbeat_interval": 5.0,
        "timeout": 60.0,
        "batch_size": 1,
        "prefetch_batches": 2,
        "auto_discover_devices": True,
        "device_list": device_configs
    }
    
    # Save configuration
    with open("distributed_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration updated with {len(device_configs)} devices")
    
    # Display cluster summary
    print("\n📊 Cluster Summary:")
    print("=" * 60)
    total_memory = sum(d['capabilities']['memory_gb'] for d in device_configs)
    total_gpu_cores = sum(d['capabilities']['gpu_cores'] for d in device_configs)
    total_cpu_cores = sum(d['capabilities']['cpu_cores'] for d in device_configs)
    
    print(f"Total Devices: {len(device_configs)}")
    print(f"Total Memory: {total_memory} GB")
    print(f"Total GPU Cores: {total_gpu_cores}")
    print(f"Total CPU Cores: {total_cpu_cores}")
    print("\nDevices:")
    for d in device_configs:
        cap = d['capabilities']
        print(f"  - {cap['device_name']}: {cap['memory_gb']}GB RAM, {cap['gpu_cores']} GPU cores")
    
    # Copy to other devices
    print("\n📤 Copying configuration to all devices...")
    subprocess.run("scp distributed_config.json mini2.local:Movies/mlx_distributed/", shell=True)
    subprocess.run("scp distributed_config.json georgedekker@master.local:~/Movies/mlx_distributed/", shell=True)
    print("✅ Configuration distributed to all devices")


if __name__ == "__main__":
    update_cluster_configuration()