#!/usr/bin/env python3
"""
Simple test of exo's sharding approach
Run each shard independently and pass tensors via files for testing
"""
import sys
import os
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from mlx_lm import load
from mlx_lm.models.qwen2 import ModelArgs

# Get rank from command line
rank = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print(f"Starting rank {rank} on {os.uname().nodename}")

# Simple shard definition
class Shard:
    def __init__(self, start, end, total):
        self.start_layer = start
        self.end_layer = end
        self.n_layers = total
    
    def is_first_layer(self):
        return self.start_layer == 0
    
    def is_last_layer(self):
        return self.end_layer == self.n_layers - 1

# Define shards
if rank == 0:
    shard = Shard(0, 13, 28)
else:
    shard = Shard(14, 27, 28)

print(f"Rank {rank}: layers {shard.start_layer}-{shard.end_layer}")

# Load model path
model_name = "mlx-community/Qwen3-1.7B-8bit"
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_id = f"models--{model_name.replace('/', '--')}"
model_path = cache_dir / model_id / "snapshots"
model_path = sorted([d for d in model_path.iterdir() if d.is_dir()])[-1]

print(f"Model path: {model_path}")

# Load config
with open(model_path / "config.json") as f:
    config = json.load(f)

# Add shard info to config
config["shard"] = {
    "model_id": model_name,
    "start_layer": shard.start_layer,
    "end_layer": shard.end_layer,
    "n_layers": shard.n_layers
}

# Import exo's sharded qwen2 model
sys.path.insert(0, "/Users/mini1/Movies/exo")
from exo.inference.mlx.models.qwen2 import Model, ModelArgs

# Create model with shard
args = ModelArgs.from_dict(config)
model = Model(args)

# Load weights
import glob
weight_files = sorted(glob.glob(str(model_path / "*.safetensors")))
weights = {}
for wf in weight_files:
    weights.update(mx.load(wf))

# Filter weights using sanitize
filtered_weights = model.sanitize(weights)
print(f"Rank {rank}: Loading {len(filtered_weights)} weight tensors")

# Load weights
model.load_weights(list(filtered_weights.items()), strict=False)
mx.eval(model.parameters())
model.eval()

print(f"Rank {rank}: Model loaded, memory: {mx.get_active_memory() / 1e9:.2f} GB")

# Test forward pass
if rank == 0:
    # Create test input
    test_input = mx.array([[1, 2, 3, 4, 5]])
    print(f"Rank 0: Input shape: {test_input.shape}")
    
    # Forward through first shard
    output = model(test_input)
    mx.eval(output)
    print(f"Rank 0: Output shape: {output.shape}, dtype: {output.dtype}")
    
    # Save output for rank 1
    np.save("/tmp/shard_output.npy", np.array(output))
    print("Rank 0: Saved output to /tmp/shard_output.npy")
    
else:
    # Wait for rank 0's output
    import time
    while not Path("/tmp/shard_output.npy").exists():
        time.sleep(0.1)
    
    # Load rank 0's output
    input_array = np.load("/tmp/shard_output.npy")
    input_tensor = mx.array(input_array)
    print(f"Rank 1: Input shape: {input_tensor.shape}")
    
    # Forward through second shard
    output = model(input_tensor)
    mx.eval(output)
    print(f"Rank 1: Output shape: {output.shape}, dtype: {output.dtype}")
    
    # Clean up
    os.remove("/tmp/shard_output.npy")