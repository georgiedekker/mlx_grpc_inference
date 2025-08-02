#!/usr/bin/env python3
"""
Fix MLX imports across all training files
"""

import os
import re
from pathlib import Path

def fix_mlx_imports(file_path):
    """Fix MLX imports in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Replace mlx.array with mx.array
        content = re.sub(r'\bmlx\.array\b', 'mx.array', content)
        
        # Replace mlx.zeros with mx.zeros
        content = re.sub(r'\bmlx\.zeros\b', 'mx.zeros', content)
        
        # Replace mlx.ones with mx.ones  
        content = re.sub(r'\bmlx\.ones\b', 'mx.ones', content)
        
        # Replace mlx.no_grad with mx.no_grad
        content = re.sub(r'\bmlx\.no_grad\b', 'mx.no_grad', content)
        
        # Replace mlx.eval with mx.eval
        content = re.sub(r'\bmlx\.eval\b', 'mx.eval', content)
        
        # Replace mlx.savez with mx.savez  
        content = re.sub(r'\bmlx\.savez\b', 'mx.savez', content)
        
        # Replace mlx.load with mx.load
        content = re.sub(r'\bmlx\.load\b', 'mx.load', content)
        
        # Replace mlx.sum with mx.sum
        content = re.sub(r'\bmlx\.sum\b', 'mx.sum', content)
        
        # Replace mlx.mean with mx.mean
        content = re.sub(r'\bmlx\.mean\b', 'mx.mean', content)
        
        # Replace mlx.sqrt with mx.sqrt
        content = re.sub(r'\bmlx\.sqrt\b', 'mx.sqrt', content)
        
        # Replace mlx.clip with mx.clip
        content = re.sub(r'\bmlx\.clip\b', 'mx.clip', content)
        
        # Replace mlx.exp with mx.exp
        content = re.sub(r'\bmlx\.exp\b', 'mx.exp', content)
        
        # Replace mlx.log with mx.log
        content = re.sub(r'\bmlx\.log\b', 'mx.log', content)
        
        # Replace mlx.minimum with mx.minimum
        content = re.sub(r'\bmlx\.minimum\b', 'mx.minimum', content)
        
        # Replace mlx.maximum with mx.maximum
        content = re.sub(r'\bmlx\.maximum\b', 'mx.maximum', content)
        
        # Replace mlx.value_and_grad with mx.value_and_grad
        content = re.sub(r'\bmlx\.value_and_grad\b', 'mx.value_and_grad', content)
        
        # Replace mlx.grad with mx.grad
        content = re.sub(r'\bmlx\.grad\b', 'mx.grad', content)
        
        # Replace mlx.random with mx.random
        content = re.sub(r'\bmlx\.random\b', 'mx.random', content)
        
        # Add import mlx.core as mx if not present and mlx references were found
        if content != original_content and 'import mlx.core as mx' not in content:
            # Check if there's already an mlx import to replace
            if re.search(r'^import mlx$', content, re.MULTILINE):
                content = re.sub(r'^import mlx$', 'import mlx.core as mx', content, flags=re.MULTILINE)
            elif 'import mlx' not in content:
                # Add the import at the top
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        lines.insert(i, 'import mlx.core as mx')
                        break
                content = '\n'.join(lines)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix MLX imports in all training files."""
    training_dir = Path("src/training")
    
    if not training_dir.exists():
        print("Training directory not found!")
        return
    
    files_fixed = 0
    total_files = 0
    
    for py_file in training_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        total_files += 1
        print(f"Processing {py_file.name}...", end=" ")
        
        if fix_mlx_imports(py_file):
            print("✅ Fixed")
            files_fixed += 1
        else:
            print("⏭️  No changes needed")
    
    print(f"\n📊 Summary: Fixed {files_fixed}/{total_files} files")

if __name__ == "__main__":
    main()