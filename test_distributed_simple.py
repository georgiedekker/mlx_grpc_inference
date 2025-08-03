#!/usr/bin/env python3
"""
Simple test to debug distributed inference step by step.
"""

import sys
sys.path.insert(0, 'src')

import mlx.core as mx
from mlx_lm import load
from core.config import ClusterConfig
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_distributed_simple():
    """Test each step of distributed inference separately."""
    print("🧪 Testing distributed inference step by step...")
    
    # Load model and config
    print("📦 Loading model and config...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    config = ClusterConfig.from_yaml('config/cluster_config.yaml')
    
    # Test input
    prompt = "user: Adaptive Multi-Teacher Multi-level Knowledge Distillation\nassistant: "
    input_ids = mx.array(tokenizer.encode(prompt))
    if len(input_ids.shape) == 1:
        input_ids = input_ids[None, :]
    
    print(f"🔤 Input tokens: {input_ids.shape} - {input_ids.tolist()[0][:10]}...")
    
    # Step 1: Embeddings (coordinator)
    print("\n🔄 Step 1: Embeddings...")
    hidden_states = model.model.embed_tokens(input_ids)
    print(f"   Shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
    print(f"   Mean: {mx.mean(hidden_states).item():.6f}")
    print(f"   First few values: {hidden_states.flatten()[:5].tolist()}")
    
    # Step 2: Coordinator layers (0-9)
    print("\n🔄 Step 2: Coordinator layers (0-9)...")
    coordinator_layers = config.model.get_device_layers(config.coordinator_device_id)
    print(f"   Processing layers: {coordinator_layers}")
    
    for layer_idx in coordinator_layers:
        layer_output = model.model.layers[layer_idx](hidden_states)
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
        else:
            hidden_states = layer_output
        
        if layer_idx in [0, 1, 8, 9]:  # Log first/last coordinator layers
            print(f"   After layer {layer_idx}: mean = {mx.mean(hidden_states).item():.6f}")
    
    print(f"   Final coordinator output:")
    print(f"   Shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")  
    print(f"   Mean: {mx.mean(hidden_states).item():.6f}")
    print(f"   First few values: {hidden_states.flatten()[:5].tolist()}")
    
    # Step 3: Simulate worker processing WITHOUT network transmission
    print("\n🔄 Step 3: Worker layers (local simulation)...")
    
    # mini2 layers (10-18)
    mini2_layers = config.model.get_device_layers("mini2")
    print(f"   mini2 layers: {mini2_layers}")
    
    for layer_idx in mini2_layers:
        layer_output = model.model.layers[layer_idx](hidden_states)
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
        else:
            hidden_states = layer_output
        
        if layer_idx in [10, 11, 17, 18]:  # Log first/last mini2 layers
            print(f"   After layer {layer_idx}: mean = {mx.mean(hidden_states).item():.6f}")
    
    # master layers (19-27)
    master_layers = config.model.get_device_layers("master")
    print(f"   master layers: {master_layers}")
    
    for layer_idx in master_layers:
        layer_output = model.model.layers[layer_idx](hidden_states)
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
        else:
            hidden_states = layer_output
        
        if layer_idx in [19, 20, 26, 27]:  # Log first/last master layers
            print(f"   After layer {layer_idx}: mean = {mx.mean(hidden_states).item():.6f}")
    
    print(f"   Final worker output:")
    print(f"   Shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
    print(f"   Mean: {mx.mean(hidden_states).item():.6f}")
    print(f"   First few values: {hidden_states.flatten()[:5].tolist()}")
    
    # Step 4: Final norm and logits
    print("\n🔄 Step 4: Final norm and logits...")
    hidden_states = model.model.norm(hidden_states)
    print(f"   After norm: mean = {mx.mean(hidden_states).item():.6f}")
    
    logits = model.model.embed_tokens.as_linear(hidden_states)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits mean: {mx.mean(logits).item():.6f}")
    
    # Step 5: Sample token
    print("\n🔄 Step 5: Sample token...")
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.1)
    next_token = sampler(logits[:, -1:, :])
    token_id = next_token.item()
    token_text = tokenizer.decode([token_id])
    
    print(f"   Next token: {token_id} -> {repr(token_text)}")
    
    # Compare with standalone model
    print("\n🔄 Step 6: Compare with standalone...")
    
    # Reset and run through model directly
    test_ids = mx.array(tokenizer.encode(prompt))
    if len(test_ids.shape) == 1:
        test_ids = test_ids[None, :]
    
    # Full forward pass
    standalone_hidden = model.model.embed_tokens(test_ids)
    for i, layer in enumerate(model.model.layers):
        standalone_hidden = layer(standalone_hidden)
        if isinstance(standalone_hidden, tuple):
            standalone_hidden = standalone_hidden[0]
    
    standalone_hidden = model.model.norm(standalone_hidden) 
    standalone_logits = model.model.embed_tokens.as_linear(standalone_hidden)
    
    standalone_token = sampler(standalone_logits[:, -1:, :])
    standalone_token_id = standalone_token.item()
    standalone_text = tokenizer.decode([standalone_token_id])
    
    print(f"   Standalone token: {standalone_token_id} -> {repr(standalone_text)}")
    
    # Compare
    logits_diff = mx.mean(mx.abs(logits - standalone_logits)).item()
    print(f"   Logits difference: {logits_diff:.8f}")
    
    if token_id == standalone_token_id:
        print("✅ Distributed processing matches standalone!")
    else:
        print("❌ Distributed processing differs from standalone!")
        print(f"   Expected: {standalone_token_id} -> {repr(standalone_text)}")
        print(f"   Got: {token_id} -> {repr(token_text)}")

if __name__ == "__main__":
    asyncio.run(test_distributed_simple())