#!/usr/bin/env python3
"""
Check which MLX models support pipeline parallelism.
This scans all downloaded models in the Hugging Face cache.
"""

import os
from pathlib import Path
from mlx_lm.utils import load_model
import mlx.core as mx

def check_pipeline_support():
    """Check all cached MLX models for pipeline() support."""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    
    print("Scanning MLX models for pipeline() support...")
    print("=" * 60)
    
    results = []
    
    # Find all model directories
    for model_dir in cache_dir.glob("models--*"):
        model_name = model_dir.name.replace("models--", "").replace("--", "/")
        
        # Skip non-MLX models
        if not model_name.startswith("mlx-community/"):
            continue
            
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
            
        # Get the first snapshot
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            continue
            
        snapshot_path = snapshots[0]
        
        # Check if it's actually an MLX model (has config.json)
        if not (snapshot_path / "config.json").exists():
            continue
            
        print(f"\nChecking: {model_name}")
        
        try:
            # Load model lazily to check structure
            model, _ = load_model(snapshot_path, lazy=True, strict=False)
            
            # Check for pipeline support
            has_pipeline = False
            model_class = None
            
            if hasattr(model, 'model'):
                model_class = model.model.__class__.__name__
                has_pipeline = hasattr(model.model, 'pipeline')
                
                # Also check if it's callable
                if has_pipeline:
                    has_pipeline = callable(getattr(model.model, 'pipeline'))
            
            status = "✅ SUPPORTED" if has_pipeline else "❌ NOT SUPPORTED"
            
            results.append({
                'name': model_name,
                'class': model_class,
                'pipeline': has_pipeline,
                'status': status
            })
            
            print(f"  Model class: {model_class}")
            print(f"  Pipeline support: {status}")
            
        except Exception as e:
            print(f"  ⚠️ Error loading model: {e}")
            results.append({
                'name': model_name,
                'class': 'Error',
                'pipeline': False,
                'status': '⚠️ ERROR'
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Pipeline Support Status")
    print("=" * 60)
    
    supported = [r for r in results if r['pipeline']]
    not_supported = [r for r in results if not r['pipeline'] and r['class'] != 'Error']
    errors = [r for r in results if r['class'] == 'Error']
    
    if supported:
        print(f"\n✅ Models WITH pipeline() support ({len(supported)}):")
        for r in supported:
            print(f"  - {r['name']} ({r['class']})")
    else:
        print("\n❌ No models found with pipeline() support")
    
    if not_supported:
        print(f"\n❌ Models WITHOUT pipeline() support ({len(not_supported)}):")
        for r in not_supported:
            print(f"  - {r['name']} ({r['class']})")
    
    if errors:
        print(f"\n⚠️ Models with errors ({len(errors)}):")
        for r in errors:
            print(f"  - {r['name']}")
    
    print("\n" + "=" * 60)
    print("To use distributed inference with pipeline parallelism,")
    print("you need a model from the '✅ SUPPORTED' list.")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    # Initialize MLX distributed (optional, just to check)
    try:
        group = mx.distributed.init()
        if group:
            print(f"MLX Distributed available: {group.size()} devices")
            print()
    except:
        pass
    
    results = check_pipeline_support()
    
    # Save results to file
    with open("pipeline_support_results.txt", "w") as f:
        f.write("MLX Models Pipeline Support Check\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"{r['name']}: {r['status']}\n")
            f.write(f"  Class: {r['class']}\n\n")
    
    print("\nResults saved to pipeline_support_results.txt")