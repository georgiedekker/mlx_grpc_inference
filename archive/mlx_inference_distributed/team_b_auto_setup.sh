#!/bin/bash

# Team B Auto Setup Script
# Automatically copies all required files for LoRA and dataset integration

set -e  # Exit on error

echo "🚀 Team B Integration Auto-Setup Script"
echo "========================================"

# Configuration
TEAM_B_DIR="/Users/mini1/Movies/mlx_distributed_training"
MLX_DIST_DIR="/Users/mini1/Movies/mlx_distributed"
BACKUP_DIR="${TEAM_B_DIR}/backup_$(date +%Y%m%d_%H%M%S)"

# Check if Team B directory exists
if [ ! -d "$TEAM_B_DIR" ]; then
    echo "❌ Team B directory not found: $TEAM_B_DIR"
    echo "Please ensure your training project is at the expected location."
    exit 1
fi

echo "✅ Found Team B directory: $TEAM_B_DIR"

# Create backup of existing files
echo "📦 Creating backup of existing files..."
mkdir -p "$BACKUP_DIR"

# Function to backup and copy file
backup_and_copy() {
    local src="$1"
    local dest="$2"
    local dest_dir=$(dirname "$dest")
    
    # Create destination directory
    mkdir -p "$dest_dir"
    
    # Backup existing file if it exists
    if [ -f "$dest" ]; then
        echo "  📋 Backing up existing: $(basename "$dest")"
        cp "$dest" "$BACKUP_DIR/"
    fi
    
    # Copy new file
    echo "  ✅ Copying: $src -> $dest"
    cp "$src" "$dest"
}

# Copy LoRA implementation
echo ""
echo "📂 Copying LoRA Components..."
echo "-----------------------------"

# Main LoRA implementation
LORA_SRC="${MLX_DIST_DIR}/mlx_knowledge_distillation/mlx_distributed_training/archived_components/lora/lora.py"
LORA_DEST="${TEAM_B_DIR}/src/mlx_distributed_training/training/lora/lora.py"

if [ -f "$LORA_SRC" ]; then
    backup_and_copy "$LORA_SRC" "$LORA_DEST"
else
    echo "❌ LoRA source not found: $LORA_SRC"
fi

# LoRA integration helper
LORA_INT_SRC="${MLX_DIST_DIR}/team_b_integration/lora/lora_integration.py"
LORA_INT_DEST="${TEAM_B_DIR}/src/mlx_distributed_training/integration/lora_integration.py"

if [ -f "$LORA_INT_SRC" ]; then
    backup_and_copy "$LORA_INT_SRC" "$LORA_INT_DEST"
else
    echo "❌ LoRA integration source not found: $LORA_INT_SRC"
fi

echo ""
echo "📂 Copying Dataset Components..."
echo "--------------------------------"

# Dataset implementations
DATASET_DIR="${MLX_DIST_DIR}/mlx_knowledge_distillation/mlx_distributed_training/archived_components/datasets"
TEAM_B_DATASET_DIR="${TEAM_B_DIR}/src/mlx_distributed_training/datasets"

if [ -d "$DATASET_DIR" ]; then
    for dataset_file in "$DATASET_DIR"/*.py; do
        if [ -f "$dataset_file" ]; then
            filename=$(basename "$dataset_file")
            backup_and_copy "$dataset_file" "$TEAM_B_DATASET_DIR/$filename"
        fi
    done
else
    echo "❌ Dataset source directory not found: $DATASET_DIR"
fi

# Dataset integration helper
DATASET_INT_SRC="${MLX_DIST_DIR}/team_b_integration/datasets/dataset_integration.py"
DATASET_INT_DEST="${TEAM_B_DIR}/src/mlx_distributed_training/integration/dataset_integration.py"

if [ -f "$DATASET_INT_SRC" ]; then
    backup_and_copy "$DATASET_INT_SRC" "$DATASET_INT_DEST"
else
    echo "❌ Dataset integration source not found: $DATASET_INT_SRC"
fi

echo ""
echo "📂 Copying Example Files..."
echo "---------------------------"

# Copy example configs and data
EXAMPLES_SRC_DIR="${MLX_DIST_DIR}/team_b_integration/examples"
EXAMPLES_DEST_DIR="${TEAM_B_DIR}/examples"

if [ -d "$EXAMPLES_SRC_DIR" ]; then
    mkdir -p "$EXAMPLES_DEST_DIR"
    cp -r "$EXAMPLES_SRC_DIR"/* "$EXAMPLES_DEST_DIR/"
    echo "  ✅ Copied example files to $EXAMPLES_DEST_DIR"
else
    echo "❌ Examples source directory not found: $EXAMPLES_SRC_DIR"
fi

# Copy sample training config
CONFIG_SRC="${MLX_DIST_DIR}/team_b_integration/sample_training_config.yaml"
CONFIG_DEST="${TEAM_B_DIR}/configs/team_b_training_config.yaml"

if [ -f "$CONFIG_SRC" ]; then
    backup_and_copy "$CONFIG_SRC" "$CONFIG_DEST"
else
    echo "❌ Sample config not found: $CONFIG_SRC"
fi

echo ""
echo "📂 Creating Integration Scripts..."
echo "---------------------------------"

# Create integration test script
cat > "${TEAM_B_DIR}/test_integration.py" << 'EOF'
#!/usr/bin/env python3
"""
Team B Integration Test Script
Tests LoRA and dataset integration after setup.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_lora_import():
    """Test LoRA implementation import."""
    try:
        from mlx_distributed_training.training.lora.lora import (
            LoRAConfig, LoRALayer, LoRALinear, apply_lora_to_model
        )
        print("✅ LoRA implementation imported successfully")
        return True
    except ImportError as e:
        print(f"❌ LoRA import failed: {e}")
        return False

def test_dataset_import():
    """Test dataset implementations import."""
    try:
        from mlx_distributed_training.datasets.alpaca_dataset import AlpacaDataset
        from mlx_distributed_training.datasets.sharegpt_dataset import ShareGPTDataset
        from mlx_distributed_training.integration.dataset_integration import validate_dataset
        print("✅ Dataset implementations imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Dataset import failed: {e}")
        return False

def test_integration_helpers():
    """Test integration helper imports."""
    try:
        from mlx_distributed_training.integration.lora_integration import (
            LoRATrainingConfig, create_lora_enabled_trainer
        )
        from mlx_distributed_training.integration.dataset_integration import (
            detect_dataset_format, validate_dataset
        )
        print("✅ Integration helpers imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Integration helpers import failed: {e}")
        return False

def test_example_files():
    """Test that example files exist."""
    base_path = Path(__file__).parent
    
    files_to_check = [
        "examples/alpaca_example.json",
        "examples/sharegpt_example.json",
        "configs/team_b_training_config.yaml"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    print("🧪 Team B Integration Test")
    print("=" * 30)
    
    tests = [
        ("LoRA Import", test_lora_import),
        ("Dataset Import", test_dataset_import),
        ("Integration Helpers", test_integration_helpers),
        ("Example Files", test_example_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append(result)
    
    print(f"\n📊 Test Results")
    print("=" * 20)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Integration setup successful.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x "${TEAM_B_DIR}/test_integration.py"
echo "  ✅ Created integration test script"

# Create quick start guide
cat > "${TEAM_B_DIR}/QUICK_START_INTEGRATION.md" << 'EOF'
# Team B Quick Start Integration Guide

## What Was Copied

This script copied the following components to your project:

### LoRA Components
- `src/mlx_distributed_training/training/lora/lora.py` - Main LoRA implementation
- `src/mlx_distributed_training/integration/lora_integration.py` - Integration helpers

### Dataset Components  
- `src/mlx_distributed_training/datasets/` - Alpaca and ShareGPT dataset loaders
- `src/mlx_distributed_training/integration/dataset_integration.py` - Dataset validation

### Examples and Config
- `examples/` - Sample dataset files
- `configs/team_b_training_config.yaml` - Sample training configuration
- `test_integration.py` - Integration test script

## Next Steps

1. **Test the integration:**
   ```bash
   cd /Users/mini1/Movies/mlx_distributed_training
   python test_integration.py
   ```

2. **Update your API** (see the detailed API modification guide)

3. **Test with sample data:**
   ```bash
   # Test dataset validation
   curl -X POST http://localhost:8200/datasets/validate \
     -H "Content-Type: application/json" \
     -d '{"file_path": "examples/alpaca_example.json"}'
   ```

4. **Start a LoRA training job:**
   ```bash
   curl -X POST http://localhost:8200/train/start \
     -H "Content-Type: application/json" \
     -d @configs/team_b_training_config.yaml
   ```

## Backup Location

Your original files were backed up to: `backup_$(date +%Y%m%d_%H%M%S)/`

## Troubleshooting

If tests fail:
1. Check that all source files exist in the mlx_distributed directory
2. Verify your Team B directory structure matches expectations
3. Check Python import paths in your project

## Support Files Created

- `team_b_api_modifications.py` - Complete API update examples
- `team_b_validation_script.py` - Validation and testing
- `team_b_complete_example.py` - End-to-end working example
EOF

echo "  ✅ Created quick start guide"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo "📍 Backup location: $BACKUP_DIR"
echo "📖 Quick start guide: ${TEAM_B_DIR}/QUICK_START_INTEGRATION.md"
echo "🧪 Test integration: cd $TEAM_B_DIR && python test_integration.py"
echo ""
echo "Next: Run the integration test and follow the API modification guide!"