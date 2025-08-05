#!/usr/bin/env python3
"""
Demo script to showcase heterogeneous model sharding capabilities.
"""
import os
import sys
import json
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.device_capability import DeviceCapability, DeviceProfiler, estimate_memory_per_layer
from utils.layer_assignment import calculate_layer_distribution
from utils.device_manager import DeviceManager


def visualize_layer_distribution(
    devices: List[DeviceCapability],
    assignments: Dict[str, any],
    model_name: str,
    total_layers: int
):
    """Create a visual representation of layer distribution."""
    print(f"\n{'='*70}")
    print(f"Model: {model_name} ({total_layers} layers)")
    print(f"{'='*70}")
    
    # Memory per layer
    mem_per_layer = estimate_memory_per_layer(model_name)
    print(f"Memory per layer: {mem_per_layer:.2f} GB")
    print()
    
    # Device capabilities
    print("Device Capabilities:")
    for device in devices:
        print(f"  {device.device_id:10} - GPU: {device.gpu_cores:2} cores, "
              f"Memory: {device.gpu_memory_gb:4.1f} GB, Score: {device.compute_score:5.1f}")
    print()
    
    # Layer assignments
    print("Layer Distribution:")
    total_memory = 0
    
    for device in devices:
        assignment = assignments[device.device_id]
        memory_used = assignment.num_layers * mem_per_layer
        total_memory += memory_used
        percentage = (assignment.num_layers / total_layers) * 100
        
        # Create visual bar
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"  {device.device_id:10} [{bar}] {percentage:5.1f}%")
        print(f"             Layers {assignment.start_layer:2d}-{assignment.end_layer:2d} "
              f"({assignment.num_layers:2d} layers), {memory_used:4.1f} GB")
        
        if assignment.has_embeddings:
            print(f"             + Embeddings")
        if assignment.has_lm_head:
            print(f"             + LM Head")
        print()
    
    print(f"Total memory usage: {total_memory:.1f} GB")
    print(f"{'='*70}\n")


def main():
    """Run heterogeneous sharding demo."""
    print("\nHeterogeneous Model Sharding Demo")
    print("=================================\n")
    
    # Demo configurations
    demos = [
        {
            "name": "Current Setup (2 Mac minis)",
            "devices": ["mini1", "mini2"],
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "layers": 28
        },
        {
            "name": "Enhanced Setup (2 Mac minis + MacBook Pro)",
            "devices": ["mini1", "mini2", "master"],
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "layers": 28
        },
        {
            "name": "Large Model (3 devices)",
            "devices": ["mini1", "mini2", "master"],
            "model": "meta-llama/Llama-2-7b",
            "layers": 32
        },
        {
            "name": "Very Large Model (4 devices)",
            "devices": ["mini1", "mini2", "macbook-pro", "master"],
            "model": "meta-llama/Llama-2-13b",
            "layers": 40
        }
    ]
    
    for demo in demos:
        print(f"\n{demo['name']}")
        print("-" * len(demo['name']))
        
        # Get device capabilities
        devices = []
        for device_name in demo['devices']:
            device = DeviceProfiler._get_predefined_profile(device_name)
            devices.append(device)
        
        # Calculate layer distribution
        assignments = calculate_layer_distribution(
            devices=devices,
            total_layers=demo['layers'],
            strategy="capability_based",
            model_name=demo['model']
        )
        
        # Visualize
        visualize_layer_distribution(
            devices=devices,
            assignments=assignments,
            model_name=demo['model'],
            total_layers=demo['layers']
        )
        
        # Show efficiency gains
        if len(devices) == 3 and "mini1" in demo['devices'] and "master" in demo['devices']:
            # Calculate comparison with equal distribution
            equal_assignments = calculate_layer_distribution(
                devices=devices,
                total_layers=demo['layers'],
                strategy="equal",
                model_name=demo['model']
            )
            
            # Find the bottleneck (slowest device)
            capability_bottleneck = min(
                devices, 
                key=lambda d: d.compute_score / assignments[d.device_id].num_layers
            )
            equal_bottleneck = min(
                devices,
                key=lambda d: d.compute_score / equal_assignments[d.device_id].num_layers
            )
            
            capability_time = assignments[capability_bottleneck.device_id].num_layers / capability_bottleneck.compute_score
            equal_time = equal_assignments[equal_bottleneck.device_id].num_layers / equal_bottleneck.compute_score
            
            speedup = equal_time / capability_time
            print(f"Performance Analysis:")
            print(f"  Equal distribution bottleneck: {equal_bottleneck.device_id}")
            print(f"  Capability-based bottleneck: {capability_bottleneck.device_id}")
            print(f"  Estimated speedup: {speedup:.2f}x faster with capability-based sharding")


if __name__ == "__main__":
    main()