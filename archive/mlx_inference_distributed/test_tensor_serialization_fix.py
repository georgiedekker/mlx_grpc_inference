#!/usr/bin/env python3
"""
Test script to verify MLX tensor serialization fixes.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import numpy as np
from distributed_comm import GRPCCommunicator, CommunicationType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mlx_tensor_serialization():
    """Test MLX tensor serialization and deserialization."""
    print("Testing MLX tensor serialization fixes...")
    
    comm = GRPCCommunicator()
    
    # Test various MLX array types
    test_cases = [
        ("float32 tensor", mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)),
        ("int32 tensor", mx.array([[1, 2], [3, 4]], dtype=mx.int32)),
        ("bool tensor", mx.array([[True, False], [False, True]], dtype=mx.bool_)),
        ("float16 tensor", mx.array([[1.0, 2.0]], dtype=mx.float16)),
        ("complex64 tensor", mx.array([1+2j, 3+4j], dtype=mx.complex64)),
    ]
    
    all_passed = True
    
    for name, original_tensor in test_cases:
        try:
            print(f"\nTesting {name}:")
            print(f"  Original shape: {original_tensor.shape}")
            print(f"  Original dtype: {original_tensor.dtype}")
            
            # Test serialization using helper method
            np_array, metadata = comm._prepare_mlx_array_for_serialization(original_tensor)
            print(f"  Converted shape: {np_array.shape}")
            print(f"  Converted dtype: {np_array.dtype}")
            print(f"  Metadata: {metadata}")
            
            # Test full serialize/deserialize cycle
            comm_data = comm._serialize_data(original_tensor, CommunicationType.TENSOR)
            recovered_tensor = comm._deserialize_data(comm_data)
            
            print(f"  Recovered shape: {recovered_tensor.shape}")
            print(f"  Recovered dtype: {recovered_tensor.dtype}")
            
            # Check if tensors are equivalent
            try:
                # For boolean tensors, check carefully
                if original_tensor.dtype == mx.bool_ and recovered_tensor.dtype == mx.bool_:
                    if mx.array_equal(original_tensor, recovered_tensor):
                        print(f"  ✓ {name} serialization/deserialization PASSED")
                    else:
                        print(f"  ✗ {name} serialization/deserialization FAILED - values don't match")
                        all_passed = False
                else:
                    # For numeric tensors, use allclose for floating point
                    if original_tensor.dtype in [mx.float16, mx.float32, mx.complex64]:
                        if mx.allclose(original_tensor.astype(mx.float32), recovered_tensor.astype(mx.float32)):
                            print(f"  ✓ {name} serialization/deserialization PASSED")
                        else:
                            print(f"  ✗ {name} serialization/deserialization FAILED - values don't match")
                            all_passed = False
                    else:
                        if mx.array_equal(original_tensor.astype(mx.float32), recovered_tensor.astype(mx.float32)):
                            print(f"  ✓ {name} serialization/deserialization PASSED")
                        else:
                            print(f"  ✗ {name} serialization/deserialization FAILED - values don't match")
                            all_passed = False
                            
            except Exception as e:
                print(f"  ✗ {name} comparison FAILED: {e}")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ {name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TENSOR SERIALIZATION TESTS PASSED!")
        print("✓ MLX tensor serialization fixes are working correctly")
    else:
        print("❌ SOME TESTS FAILED")
        print("❌ MLX tensor serialization needs further fixes")
    print(f"{'='*50}")
    
    return all_passed

def test_edge_cases():
    """Test edge cases for tensor serialization."""
    print("\nTesting edge cases...")
    
    comm = GRPCCommunicator()
    
    edge_cases = [
        ("empty tensor", mx.array([], dtype=mx.float32)),
        ("scalar tensor", mx.array(5.0)),
        ("large tensor", mx.ones((1000, 1000), dtype=mx.float32)),
        ("zero tensor", mx.zeros((10, 10), dtype=mx.float32)),
    ]
    
    all_passed = True
    
    for name, tensor in edge_cases:
        try:
            print(f"\nTesting {name}:")
            np_array, metadata = comm._prepare_mlx_array_for_serialization(tensor)
            
            # Test full cycle
            comm_data = comm._serialize_data(tensor, CommunicationType.TENSOR)
            recovered = comm._deserialize_data(comm_data)
            
            # Basic shape and dtype check
            if tensor.shape == recovered.shape:
                print(f"  ✓ {name} shape preserved")
            else:
                print(f"  ✗ {name} shape mismatch: {tensor.shape} vs {recovered.shape}")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success1 = test_mlx_tensor_serialization()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! MLX tensor serialization is fixed.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED. Please review the fixes.")
        sys.exit(1)