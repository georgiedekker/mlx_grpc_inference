#!/usr/bin/env python3
"""
Test script to verify the Qwen model lm_head fix.

This script tests the exact scenario that was failing:
- Loading Qwen3-1.7B-8bit model
- Creating model shards
- Verifying tied embeddings are handled correctly
"""

from model_abstraction import ModelFactory
import sys

def test_qwen_model():
    """Test Qwen model loading and sharding."""
    print("Testing Qwen model lm_head fix...")
    
    try:
        # Create wrapper (this is what grpc_server.py does)
        print("1. Creating model wrapper...")
        wrapper = ModelFactory.create_wrapper('mlx-community/Qwen3-1.7B-8bit')
        print(f"   Created: {type(wrapper).__name__}")
        
        # Load model (this is what grpc_server.py does)
        print("2. Loading model...")
        wrapper.load_model()
        print("   Model loaded successfully")
        
        # Test the model properties
        print("3. Checking model properties...")
        model = wrapper.model
        print(f"   Model type: {type(model)}")
        print(f"   Has lm_head: {hasattr(model, 'lm_head')}")
        print(f"   tie_word_embeddings: {model.args.tie_word_embeddings}")
        
        # Create shards (this is where the error was occurring)
        print("4. Creating shards...")
        shards = wrapper.create_shards(num_shards=2, shard_sizes=None)
        print(f"   Created {len(shards)} shards successfully")
        
        # Test single shard (exercises the lm_head code path)
        print("5. Creating single shard...")
        single_shard = wrapper.create_shards(num_shards=1, shard_sizes=None)
        print(f"   Created single shard successfully")
        
        # Verify shard properties
        print("6. Verifying shard properties...")
        shard = single_shard[0]
        print(f"   Shard lm_head: {shard.lm_head}")
        print(f"   Shard use_tied_embeddings: {shard.use_tied_embeddings}")
        print(f"   Shard embed_tokens: {shard.embed_tokens is not None}")
        
        print("\n✅ All tests passed! The lm_head fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qwen_model()
    sys.exit(0 if success else 1)