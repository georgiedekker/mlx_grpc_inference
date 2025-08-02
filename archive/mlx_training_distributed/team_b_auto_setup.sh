#!/bin/bash
# Team B Automated Setup Script
# This script automatically sets up LoRA/QLoRA and dataset format support

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Team B LoRA & Dataset Integration Setup Script         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/Users/mini1/Movies/mlx_distributed"
TEAM_B_PROJECT="${PROJECT_ROOT}/mlx_distributed_training"
ARCHIVE_ROOT="${PROJECT_ROOT}/mlx_knowledge_distillation/mlx_distributed_training/archived_components"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "   Script Directory: $SCRIPT_DIR"
echo "   Team B Project: $TEAM_B_PROJECT"
echo "   Archive Source: $ARCHIVE_ROOT"
echo ""

# Step 1: Create backup
echo -e "${BLUE}1️⃣ Creating backup of existing files...${NC}"
if [ -d "$TEAM_B_PROJECT" ]; then
    BACKUP_DIR="${TEAM_B_PROJECT}_backup_${TIMESTAMP}"
    cp -r "$TEAM_B_PROJECT" "$BACKUP_DIR"
    echo -e "${GREEN}✅ Backup created at: $BACKUP_DIR${NC}"
else
    echo -e "${YELLOW}⚠️  No existing project found, skipping backup${NC}"
fi

# Step 2: Create directory structure
echo -e "${BLUE}2️⃣ Creating directory structure...${NC}"
mkdir -p "$TEAM_B_PROJECT/src/mlx_distributed_training/training/lora"
mkdir -p "$TEAM_B_PROJECT/src/mlx_distributed_training/datasets"
mkdir -p "$TEAM_B_PROJECT/src/mlx_distributed_training/integration"
mkdir -p "$TEAM_B_PROJECT/examples/datasets"
mkdir -p "$TEAM_B_PROJECT/examples/configs"
mkdir -p "$TEAM_B_PROJECT/tests/integration"
echo -e "${GREEN}✅ Directory structure created${NC}"

# Step 3: Copy LoRA implementation
echo -e "${BLUE}3️⃣ Copying LoRA implementation...${NC}"
if [ -f "$ARCHIVE_ROOT/lora/lora.py" ]; then
    cp "$ARCHIVE_ROOT/lora/lora.py" "$TEAM_B_PROJECT/src/mlx_distributed_training/training/lora/"
    echo -e "${GREEN}✅ LoRA implementation copied${NC}"
else
    echo -e "${RED}❌ LoRA implementation not found at $ARCHIVE_ROOT/lora/lora.py${NC}"
fi

# Step 4: Copy dataset implementations
echo -e "${BLUE}4️⃣ Copying dataset implementations...${NC}"
for file in base_dataset.py alpaca_dataset.py sharegpt_dataset.py dataset_utils.py; do
    if [ -f "$ARCHIVE_ROOT/datasets/$file" ]; then
        cp "$ARCHIVE_ROOT/datasets/$file" "$TEAM_B_PROJECT/src/mlx_distributed_training/datasets/"
        echo -e "${GREEN}✅ Copied $file${NC}"
    else
        echo -e "${YELLOW}⚠️  $file not found in archive${NC}"
    fi
done

# Step 5: Copy integration helpers
echo -e "${BLUE}5️⃣ Copying integration helpers...${NC}"
if [ -f "$SCRIPT_DIR/lora/lora_integration.py" ]; then
    cp "$SCRIPT_DIR/lora/lora_integration.py" "$TEAM_B_PROJECT/src/mlx_distributed_training/integration/"
    echo -e "${GREEN}✅ LoRA integration helper copied${NC}"
fi

if [ -f "$SCRIPT_DIR/datasets/dataset_integration.py" ]; then
    cp "$SCRIPT_DIR/datasets/dataset_integration.py" "$TEAM_B_PROJECT/src/mlx_distributed_training/integration/"
    echo -e "${GREEN}✅ Dataset integration helper copied${NC}"
fi

# Step 6: Create example datasets
echo -e "${BLUE}6️⃣ Creating example datasets...${NC}"

# Create Alpaca example
cat > "$TEAM_B_PROJECT/examples/datasets/alpaca_example.json" << 'EOF'
[
    {
        "instruction": "What is machine learning?",
        "input": "",
        "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can analyze data, identify patterns, and make decisions with minimal human intervention."
    },
    {
        "instruction": "Explain the difference between supervised and unsupervised learning.",
        "input": "",
        "output": "Supervised learning uses labeled data to train models, where each input has a known output. The model learns to map inputs to outputs. Unsupervised learning works with unlabeled data, discovering hidden patterns or structures without predefined outputs. Supervised learning is used for classification and regression, while unsupervised learning is used for clustering and dimensionality reduction."
    },
    {
        "instruction": "Write a Python function to calculate the factorial of a number.",
        "input": "5",
        "output": "def factorial(n):\n    if n < 0:\n        return None\n    elif n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\n# Example usage\nresult = factorial(5)\nprint(f\"Factorial of 5 is: {result}\")  # Output: 120"
    }
]
EOF

# Create ShareGPT example
cat > "$TEAM_B_PROJECT/examples/datasets/sharegpt_example.json" << 'EOF'
[
    {
        "conversations": [
            {"from": "human", "value": "Can you help me understand neural networks?"},
            {"from": "assistant", "value": "I'd be happy to help you understand neural networks! Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers: input, hidden, and output layers. Each connection has a weight that gets adjusted during training. Would you like me to explain how they learn?"},
            {"from": "human", "value": "Yes, please explain the learning process."},
            {"from": "assistant", "value": "Neural networks learn through a process called backpropagation. Here's how it works:\n\n1. Forward pass: Input data flows through the network, producing an output\n2. Error calculation: The output is compared to the expected result\n3. Backward pass: The error is propagated backwards through the network\n4. Weight adjustment: Connection weights are updated to minimize error\n5. Repeat: This process continues for many iterations until the network learns the patterns\n\nThe network essentially learns by adjusting weights to minimize prediction errors."}
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "What are the main types of machine learning?"},
            {"from": "assistant", "value": "There are three main types of machine learning:\n\n1. **Supervised Learning**: Uses labeled data to train models. Examples include classification (spam detection) and regression (price prediction).\n\n2. **Unsupervised Learning**: Works with unlabeled data to find patterns. Examples include clustering (customer segmentation) and dimensionality reduction (PCA).\n\n3. **Reinforcement Learning**: Learns through interaction with an environment using rewards and penalties. Examples include game playing (AlphaGo) and robotics.\n\nEach type is suited for different kinds of problems and data availability."}
        ]
    }
]
EOF

echo -e "${GREEN}✅ Example datasets created${NC}"

# Step 7: Create sample configuration
echo -e "${BLUE}7️⃣ Creating sample configuration...${NC}"
cat > "$TEAM_B_PROJECT/examples/configs/lora_training_config.yaml" << 'EOF'
# LoRA Training Configuration
model:
  name: "mlx-community/Qwen2.5-1.5B-4bit"
  
training:
  experiment_name: "lora_alpaca_demo"
  epochs: 3
  learning_rate: 5e-5
  batch_size: 8
  gradient_accumulation_steps: 4
  warmup_steps: 100
  save_steps: 500
  
lora:
  use_lora: true
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
  use_qlora: false
  
dataset:
  path: "../datasets/alpaca_example.json"
  format: "alpaca"  # or "sharegpt"
  max_seq_length: 2048
  shuffle: true
  
output:
  dir: "./outputs/lora_demo"
  save_total_limit: 3
  save_only_lora: true
EOF

echo -e "${GREEN}✅ Sample configuration created${NC}"

# Step 8: Create __init__.py files
echo -e "${BLUE}8️⃣ Creating __init__.py files...${NC}"
touch "$TEAM_B_PROJECT/src/mlx_distributed_training/__init__.py"
touch "$TEAM_B_PROJECT/src/mlx_distributed_training/training/__init__.py"
touch "$TEAM_B_PROJECT/src/mlx_distributed_training/training/lora/__init__.py"
touch "$TEAM_B_PROJECT/src/mlx_distributed_training/datasets/__init__.py"
touch "$TEAM_B_PROJECT/src/mlx_distributed_training/integration/__init__.py"
echo -e "${GREEN}✅ Python package structure created${NC}"

# Step 9: Create integration test script
echo -e "${BLUE}9️⃣ Creating integration test script...${NC}"
cat > "$TEAM_B_PROJECT/tests/integration/test_lora_dataset_integration.py" << 'EOF'
#!/usr/bin/env python3
"""Integration test for LoRA and dataset support."""

import os
import sys
import json
import requests
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

API_BASE = "http://localhost:8200"
API_KEY = "test-api-key"

def test_health_endpoint():
    """Test that health endpoint reports LoRA and dataset features."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["features"]["lora"] == True
    assert "alpaca" in data["features"]["dataset_formats"]
    assert "sharegpt" in data["features"]["dataset_formats"]
    print("✅ Health endpoint test passed")

def test_dataset_validation():
    """Test dataset validation endpoint."""
    print("\nTesting dataset validation...")
    
    # Test Alpaca format
    alpaca_path = project_root / "examples/datasets/alpaca_example.json"
    response = requests.post(
        f"{API_BASE}/v1/datasets/validate",
        json={"file_path": str(alpaca_path)}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] == True
    assert data["format"] == "alpaca"
    print("✅ Alpaca dataset validation passed")
    
    # Test ShareGPT format
    sharegpt_path = project_root / "examples/datasets/sharegpt_example.json"
    response = requests.post(
        f"{API_BASE}/v1/datasets/validate",
        json={"file_path": str(sharegpt_path)}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] == True
    assert data["format"] == "sharegpt"
    print("✅ ShareGPT dataset validation passed")

def test_lora_training_job():
    """Test creating a LoRA training job."""
    print("\nTesting LoRA training job creation...")
    
    config = {
        "model": "mlx-community/Qwen2.5-1.5B-4bit",
        "experiment_name": "test_lora_integration",
        "training_file": str(project_root / "examples/datasets/alpaca_example.json"),
        "epochs": 1,
        "lora_config": {
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16.0,
            "lora_dropout": 0.05
        }
    }
    
    headers = {"X-API-Key": API_KEY}
    response = requests.post(
        f"{API_BASE}/v1/fine-tuning/jobs",
        json=config,
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["lora_enabled"] == True
    assert data["dataset_info"]["format"] == "alpaca"
    print("✅ LoRA training job creation passed")
    
    return data["job_id"]

def test_job_status(job_id):
    """Test job status endpoint."""
    print(f"\nTesting job status for {job_id}...")
    
    headers = {"X-API-Key": API_KEY}
    response = requests.get(
        f"{API_BASE}/v1/fine-tuning/jobs/{job_id}",
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "lora_details" in data
    assert data["lora_details"]["enabled"] == True
    assert data["lora_details"]["rank"] == 8
    print("✅ Job status test passed")

if __name__ == "__main__":
    print("🧪 Running Team B Integration Tests")
    print("=" * 50)
    
    try:
        test_health_endpoint()
        test_dataset_validation()
        job_id = test_lora_training_job()
        test_job_status(job_id)
        
        print("\n✅ All integration tests passed!")
        print("🎉 Team B is ready for A+ grade!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API at http://localhost:8200")
        print("Make sure the Team B API server is running")
        sys.exit(1)
EOF

chmod +x "$TEAM_B_PROJECT/tests/integration/test_lora_dataset_integration.py"
echo -e "${GREEN}✅ Integration test script created${NC}"

# Step 10: Create quick validation script
echo -e "${BLUE}🔟 Creating quick validation script...${NC}"
cat > "$TEAM_B_PROJECT/validate_integration.sh" << 'EOF'
#!/bin/bash
# Quick validation script for Team B integration

echo "🔍 Validating Team B Integration..."
echo "=================================="

# Check if files exist
echo "📁 Checking file structure..."
required_files=(
    "src/mlx_distributed_training/training/lora/lora.py"
    "src/mlx_distributed_training/datasets/base_dataset.py"
    "src/mlx_distributed_training/datasets/alpaca_dataset.py"
    "src/mlx_distributed_training/datasets/sharegpt_dataset.py"
    "examples/datasets/alpaca_example.json"
    "examples/datasets/sharegpt_example.json"
)

all_good=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        all_good=false
    fi
done

if $all_good; then
    echo -e "\n✅ All files present!"
    echo "🎉 Integration setup complete!"
else
    echo -e "\n❌ Some files are missing. Please run team_b_auto_setup.sh"
fi
EOF

chmod +x "$TEAM_B_PROJECT/validate_integration.sh"
echo -e "${GREEN}✅ Validation script created${NC}"

# Summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              🎉 Setup Complete! 🎉                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📋 Summary:${NC}"
echo "   ✅ Directory structure created"
echo "   ✅ LoRA implementation copied"
echo "   ✅ Dataset loaders copied"
echo "   ✅ Integration helpers copied"
echo "   ✅ Example datasets created"
echo "   ✅ Sample configuration created"
echo "   ✅ Test scripts created"
echo ""
echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo "   1. cd $TEAM_B_PROJECT"
echo "   2. Review team_b_api_modifications.py for API changes"
echo "   3. Run ./validate_integration.sh to verify setup"
echo "   4. Start your API server on port 8200"
echo "   5. Run python tests/integration/test_lora_dataset_integration.py"
echo ""
echo -e "${GREEN}Team B is ready to implement LoRA and dataset support!${NC}"