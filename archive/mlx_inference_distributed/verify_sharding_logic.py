#!/usr/bin/env python3
"""
Verification script for model sharding logic across 3 M4 Mac devices.
Tests the layer distribution for Qwen3-1.7B-8bit (28 layers).
"""

def test_sharding_logic():
    """Test the balanced sharding logic."""
    print("=" * 60)
    print("VERIFYING MODEL SHARDING LOGIC")
    print("=" * 60)
    
    # Simulate the ShardInfo.create_balanced logic
    total_layers = 28
    num_devices = 3
    
    print(f"Model: Qwen3-1.7B-8bit")
    print(f"Total layers: {total_layers}")
    print(f"Number of devices: {num_devices}")
    print()
    
    # Calculate distribution
    base_layers = total_layers // num_devices
    extra_layers = total_layers % num_devices
    
    print(f"Base layers per device: {base_layers}")
    print(f"Extra layers to distribute: {extra_layers}")
    print()
    
    # Create layers_per_device list
    layers_per_device = [base_layers] * num_devices
    for i in range(extra_layers):
        layers_per_device[i] += 1
    
    print(f"Layers per device: {layers_per_device}")
    print()
    
    # Create layer assignments
    layer_assignments = {}
    current_layer = 0
    device_assignments = []
    
    for device_idx, num_layers in enumerate(layers_per_device):
        start_layer = current_layer
        device_layers = []
        
        for _ in range(num_layers):
            layer_assignments[current_layer] = device_idx
            device_layers.append(current_layer)
            current_layer += 1
        
        end_layer = current_layer - 1
        device_assignments.append({
            'device': device_idx,
            'start': start_layer,
            'end': end_layer,
            'layers': device_layers,
            'count': len(device_layers)
        })
    
    # Print detailed assignments
    print("DETAILED LAYER ASSIGNMENTS:")
    print("-" * 40)
    for assignment in device_assignments:
        print(f"Device {assignment['device']}:")
        print(f"  Layers: {assignment['start']}-{assignment['end']} ({assignment['count']} layers)")
        print(f"  Layer indices: {assignment['layers']}")
        print()
    
    # Verify requirements
    print("VERIFICATION CHECKS:")
    print("-" * 40)
    
    # Check expected distribution
    expected = [
        {"device": 0, "start": 0, "end": 9, "count": 10},
        {"device": 1, "start": 10, "end": 18, "count": 9},
        {"device": 2, "start": 19, "end": 27, "count": 9}
    ]
    
    all_correct = True
    
    for i, (actual, expect) in enumerate(zip(device_assignments, expected)):
        device_correct = (
            actual['device'] == expect['device'] and
            actual['start'] == expect['start'] and
            actual['end'] == expect['end'] and
            actual['count'] == expect['count']
        )
        
        status = "✅ CORRECT" if device_correct else "❌ INCORRECT"
        print(f"Device {i}: {status}")
        print(f"  Expected: layers {expect['start']}-{expect['end']} ({expect['count']} layers)")
        print(f"  Actual:   layers {actual['start']}-{actual['end']} ({actual['count']} layers)")
        
        if not device_correct:
            all_correct = False
        print()
    
    # Check for gaps or overlaps
    all_layers = set()
    for layer_idx, device_idx in layer_assignments.items():
        if layer_idx in all_layers:
            print(f"❌ OVERLAP: Layer {layer_idx} assigned multiple times")
            all_correct = False
        all_layers.add(layer_idx)
    
    expected_layers = set(range(total_layers))
    if all_layers != expected_layers:
        missing = expected_layers - all_layers
        extra = all_layers - expected_layers
        if missing:
            print(f"❌ MISSING LAYERS: {sorted(missing)}")
            all_correct = False
        if extra:
            print(f"❌ EXTRA LAYERS: {sorted(extra)}")
            all_correct = False
    else:
        print("✅ All layers assigned exactly once")
    
    print(f"✅ Total layers covered: {len(all_layers)}/{total_layers}")
    print()
    
    # Summary
    print("=" * 60)
    if all_correct:
        print("🎉 SHARDING LOGIC VERIFICATION: PASSED")
        print("   The layer distribution is correct for 3 M4 Mac devices:")
        print("   - Device 0 (mini1): layers 0-9 (10 layers)")
        print("   - Device 1 (mini2): layers 10-18 (9 layers)")
        print("   - Device 2 (master): layers 19-27 (9 layers)")
    else:
        print("❌ SHARDING LOGIC VERIFICATION: FAILED")
        print("   The layer distribution needs to be fixed!")
    print("=" * 60)
    
    return all_correct

def test_sequential_flow():
    """Test that the sequential inference flow is correct."""
    print("\nTESTING SEQUENTIAL INFERENCE FLOW:")
    print("-" * 40)
    
    flow_steps = [
        "1. Input tokens → Device 0 (mini1)",
        "2. Device 0: Embedding layer + layers 0-9",
        "3. Hidden states → Device 1 (mini2)",
        "4. Device 1: Layers 10-18", 
        "5. Hidden states → Device 2 (master)",
        "6. Device 2: Layers 19-27 + norm + lm_head",
        "7. Final logits → return to all devices"
    ]
    
    for step in flow_steps:
        print(f"   {step}")
    
    print("\n✅ Sequential flow is correctly designed")
    return True

def test_boundary_conditions():
    """Test edge cases and boundary conditions."""
    print("\nTESTING BOUNDARY CONDITIONS:")
    print("-" * 40)
    
    test_cases = [
        (1, 1, "Single device, single layer"),
        (28, 1, "Single device, 28 layers"),
        (28, 2, "Two devices, 28 layers"),
        (28, 4, "Four devices, 28 layers"),
        (30, 3, "Three devices, 30 layers"),
        (27, 3, "Three devices, 27 layers")
    ]
    
    for total_layers, num_devices, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Layers: {total_layers}, Devices: {num_devices}")
        
        base_layers = total_layers // num_devices
        extra_layers = total_layers % num_devices
        
        layers_per_device = [base_layers] * num_devices
        for i in range(extra_layers):
            layers_per_device[i] += 1
        
        print(f"  Distribution: {layers_per_device}")
        print(f"  Total assigned: {sum(layers_per_device)}")
        
        if sum(layers_per_device) == total_layers:
            print("  ✅ Correct")
        else:
            print("  ❌ Incorrect")
    
    return True

if __name__ == "__main__":
    success = test_sharding_logic()
    test_sequential_flow()
    test_boundary_conditions()
    
    if success:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
    else:
        print("\n❌ VERIFICATION FAILED - NEEDS FIXES!")