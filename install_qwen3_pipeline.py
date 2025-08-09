#!/usr/bin/env python3
"""
Install our Qwen3 model with pipeline support into MLX-LM
This patches the existing installation to add our custom model
"""

import shutil
import sys
from pathlib import Path

def install_qwen3_pipeline():
    """Install our custom Qwen3 model with pipeline support"""
    
    # Find MLX-LM installation
    import mlx_lm
    mlx_lm_path = Path(mlx_lm.__file__).parent
    models_path = mlx_lm_path / "models"
    
    print(f"MLX-LM path: {mlx_lm_path}")
    print(f"Models path: {models_path}")
    
    # Copy our custom model
    source = Path("mlx_lm_models_qwen3_pipeline.py")
    if not source.exists():
        print(f"Error: {source} not found!")
        return False
    
    dest = models_path / "qwen3_pipeline.py"
    
    print(f"Installing {source} -> {dest}")
    shutil.copy2(source, dest)
    
    # Also create a model mapping
    # We need to modify the model type mapping to use our pipeline version
    mapping_file = models_path / "__init__.py"
    
    # Read existing init file
    with open(mapping_file, 'r') as f:
        content = f.read()
    
    # Check if we already added our mapping
    if "qwen3_pipeline" not in content:
        print("Adding model mapping...")
        
        # Add import
        import_line = "from . import qwen3_pipeline"
        if import_line not in content:
            # Find where other imports are and add ours
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("from . import"):
                    lines.insert(i+1, import_line)
                    break
            content = '\n'.join(lines)
            
            with open(mapping_file, 'w') as f:
                f.write(content)
    
    print("✅ Installation complete!")
    print("\nTo use the pipeline-enabled Qwen3 model:")
    print("1. Set model_type to 'qwen3_pipeline' in config.json")
    print("2. Or use our test script to load it directly")
    
    return True


def test_installation():
    """Test that our model can be imported"""
    try:
        from mlx_lm.models import qwen3_pipeline
        print("✅ Model imported successfully!")
        
        # Check that it has pipeline method
        import inspect
        if hasattr(qwen3_pipeline.Qwen3Model, 'pipeline'):
            print("✅ Pipeline method found!")
        else:
            print("❌ Pipeline method not found")
            
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


if __name__ == "__main__":
    if install_qwen3_pipeline():
        print("\nTesting installation...")
        test_installation()
    else:
        print("Installation failed!")
        sys.exit(1)