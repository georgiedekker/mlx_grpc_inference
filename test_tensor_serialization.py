#!/usr/bin/env python3
"""
Test tensor serialization to identify the data corruption issue.
"""

import sys
sys.path.insert(0, 'src')

import mlx.core as mx
from mlx_lm import load
from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array

def test_tensor_serialization():
    print("🧪 Testing tensor serialization for data corruption...")
    
    # Load model to get real tensors
    print("📦 Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Test with a real model tensor
    print("🔍 Testing with model embedding tensor...")
    
    # Get embeddings for a test input
    test_input = mx.array([[872, 25, 86408, 17439, 12]])  # First few tokens
    original_hidden = model.model.embed_tokens(test_input)
    
    print(f"📊 Original tensor:")
    print(f"   Shape: {original_hidden.shape}")
    print(f"   Dtype: {original_hidden.dtype}")
    print(f"   Mean: {mx.mean(original_hidden).item():.6f}")
    print(f"   First few values: {original_hidden.flatten()[:5].tolist()}")
    
    # Serialize and deserialize
    print("🔄 Serializing...")
    serialized_data, metadata = serialize_mlx_array(original_hidden)
    print(f"   Metadata: {metadata}")
    
    print("🔄 Deserializing...")
    reconstructed_hidden = deserialize_mlx_array(serialized_data, metadata)
    
    print(f"📊 Reconstructed tensor:")
    print(f"   Shape: {reconstructed_hidden.shape}")
    print(f"   Dtype: {reconstructed_hidden.dtype}")
    print(f"   Mean: {mx.mean(reconstructed_hidden).item():.6f}")
    print(f"   First few values: {reconstructed_hidden.flatten()[:5].tolist()}")
    
    # Check for differences
    print("🔍 Comparing tensors...")
    
    # Convert both to same dtype for comparison if needed
    if original_hidden.dtype != reconstructed_hidden.dtype:
        print(f"⚠️  DTYPE MISMATCH: {original_hidden.dtype} vs {reconstructed_hidden.dtype}")
        
        # Convert reconstructed back to original dtype
        print("🔧 Converting reconstructed tensor back to original dtype...")
        reconstructed_fixed = reconstructed_hidden.astype(original_hidden.dtype)
        
        print(f"📊 Fixed reconstructed tensor:")
        print(f"   Shape: {reconstructed_fixed.shape}")
        print(f"   Dtype: {reconstructed_fixed.dtype}")
        print(f"   Mean: {mx.mean(reconstructed_fixed).item():.6f}")
        print(f"   First few values: {reconstructed_fixed.flatten()[:5].tolist()}")
        
        # Check difference
        diff = mx.abs(original_hidden - reconstructed_fixed)
        max_diff = mx.max(diff).item()
        mean_diff = mx.mean(diff).item()
        
        print(f"📊 Difference after dtype fix:")
        print(f"   Max difference: {max_diff:.8f}")
        print(f"   Mean difference: {mean_diff:.8f}")
        
        if max_diff < 1e-6:
            print("✅ Tensors match after dtype correction!")
        else:
            print("❌ Tensors still differ significantly")
    else:
        # Direct comparison
        diff = mx.abs(original_hidden - reconstructed_hidden)
        max_diff = mx.max(diff).item()
        mean_diff = mx.mean(diff).item()
        
        print(f"📊 Difference:")
        print(f"   Max difference: {max_diff:.8f}")
        print(f"   Mean difference: {mean_diff:.8f}")
        
        if max_diff < 1e-6:
            print("✅ Tensors match perfectly!")
        else:
            print("❌ Tensors differ significantly")

if __name__ == "__main__":
    test_tensor_serialization()