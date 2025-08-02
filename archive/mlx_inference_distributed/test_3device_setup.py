#!/usr/bin/env python3
"""Test the 3-device configuration setup."""

import json
import os
from distributed_config import DistributedConfig

# Load the configuration
config_path = "distributed_config.json"
if os.path.exists(config_path):
    config = DistributedConfig.load(config_path)
else:
    config = DistributedConfig()

print(f"Model parallel size: {config.model_parallel_size}")
print(f"Number of devices in config: {len(config.device_list)}")
print("\nDevices:")
for device in config.device_list:
    print(f"  - Rank {device.device_index}: {device.device_id} at {device.hostname}")

# Check if JSON file has correct device count
with open("distributed_config.json", "r") as f:
    json_config = json.load(f)
    print(f"\nJSON file has {len(json_config.get('devices', []))} devices")