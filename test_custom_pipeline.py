#!/usr/bin/env python3
"""
Test our custom Qwen3 model with native pipeline support
"""

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models import qwen3_pipeline
import json
from pathlib import Path
import shutil


def setup_custom_model():
    """
    Modify the cached model to use our pipeline version
    """
    # Find the cached model
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_id = "models--mlx-community--Qwen3-1.7B-8bit"
    model_path = cache_dir / model_id
    
    # Find snapshot directory
    snapshots = model_path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
        if snapshot_dirs:
            model_path = snapshot_dirs[-1]
    
    print(f"Model path: {model_path}")
    
    # Backup original config
    config_path = model_path / "config.json"
    backup_path = model_path / "config_original.json"
    
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"Backed up config to {backup_path}")
    
    # Modify config to use our pipeline model
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Change model type to use our pipeline version
    config['model_type'] = 'qwen3_pipeline'
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Modified config to use qwen3_pipeline model type")
    
    return model_path


def main():
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("No distributed group. Run with: mpirun -n 2 python test_custom_pipeline.py")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    def rprint(*args, **kwargs):
        """Print only from rank 0"""
        if rank == 0:
            print(*args, **kwargs)
    
    rprint(f"Testing custom Qwen3 with native pipeline support")
    rprint(f"Using {world_size} GPUs")
    
    # Setup custom model
    if rank == 0:
        model_path = setup_custom_model()
    
    # Synchronize
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    # Load model (it should now use our qwen3_pipeline)
    rprint("Loading model with native pipeline support...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Check that we got our custom model
    if hasattr(model.model, 'pipeline'):
        rprint("✅ Model has pipeline method!")
    else:
        rprint("❌ Model doesn't have pipeline method")
        return
    
    # Apply pipeline
    model.model.pipeline(group)
    
    # Evaluate parameters
    mx.eval(model.parameters())
    
    # Synchronize
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    rprint("✅ Model loaded and pipeline applied!")
    rprint(f"   Rank 0: layers {model.model.start_idx}-{model.model.end_idx-1}")
    if world_size > 1:
        rprint(f"   Rank 1: layers 0-{model.model.start_idx-1}")
    
    # Test generation
    prompt = "What is 2+2? The answer is"
    rprint(f"\nPrompt: {prompt}")
    rprint("Response: ", end="", flush=True)
    
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=20
    ):
        rprint(response.text, end="", flush=True)
    
    rprint()
    
    if rank == 0:
        rprint("\n" + "="*50)
        rprint(f"Generation: {response.generation_tokens} tokens")
        rprint(f"Speed: {response.generation_tps:.1f} tokens/sec")
        rprint(f"Memory: {response.peak_memory:.2f} GB")
        rprint("="*50)
        rprint("✅ SUCCESS! Native pipeline working!")


if __name__ == "__main__":
    main()