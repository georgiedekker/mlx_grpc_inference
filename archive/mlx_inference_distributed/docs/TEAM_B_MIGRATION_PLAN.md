# Team B Migration Plan: Move to Dedicated Directory

## Current Problem
Team B is modifying core files in `mlx_distributed/` that Team A needs:
- ❌ Modified `distributed_api.py` (Team A's file!)
- ❌ Modified `distributed_config.py` (shared file)
- ❌ Creating files in root `src/mlx_distributed/`
- ❌ Risk of breaking Team A's infrastructure work

## Migration Plan to `mlx_distributed_training/`

### Step 1: Create New Directory Structure (30 minutes)
```bash
# Create parallel directory
cd /Users/mini1/Movies
mkdir -p mlx_distributed_training

# Create proper structure
cd mlx_distributed_training
mkdir -p src/mlx_distributed_training/{core,api,cli,configs,examples,tests}
mkdir -p src/mlx_distributed_training/training/{distributed,sft,optimizers,datasets,utils}

# Initialize as Python package
touch src/mlx_distributed_training/__init__.py
touch src/mlx_distributed_training/training/__init__.py
```

### Step 2: Create Independent pyproject.toml (15 minutes)
```toml
# mlx_distributed_training/pyproject.toml
[project]
name = "mlx-distributed-training"
version = "0.1.0"
description = "Distributed training framework for MLX models"
readme = "README.md"
requires-python = ">=3.13"
license = { text = "MIT" }

dependencies = [
    # Core MLX dependencies
    "mlx>=0.22.0",
    "mlx-lm>=0.21.0",
    
    # Training specific
    "datasets>=3.2.0",
    "transformers>=4.48.0",
    "wandb>=0.19.1",
    "tensorboardX>=2.6.2",
    "scikit-learn>=1.6.0",
    
    # Import mlx_distributed as dependency!
    "mlx-distributed @ file:///../mlx_distributed",
    
    # Other deps...
]

[project.scripts]
mlx-train = "mlx_distributed_training.cli.train:main"
mlx-train-serve = "mlx_distributed_training.api.server:main"
```

### Step 3: Move Team B's Files (1 hour)

#### Files to MOVE (not copy):
```bash
# From mlx_distributed to mlx_distributed_training
mv /Users/mini1/Movies/mlx_distributed/src/mlx_distributed/training/* \
   /Users/mini1/Movies/mlx_distributed_training/src/mlx_distributed_training/training/

# Move Team B's API work
mv /Users/mini1/Movies/mlx_distributed/src/mlx_distributed/api/server.py \
   /Users/mini1/Movies/mlx_distributed_training/src/mlx_distributed_training/api/

# Move any training configs
mv /Users/mini1/Movies/mlx_distributed/test_training_config.json \
   /Users/mini1/Movies/mlx_distributed_training/configs/
```

#### Files to REVERT in mlx_distributed:
```bash
# Team A's files that Team B modified - restore from backup or git
# distributed_api.py
# distributed_config.py
```

### Step 4: Create Integration Layer (2 hours)

Instead of modifying Team A's files, create adapters:

```python
# mlx_distributed_training/src/mlx_distributed_training/core/integration.py
"""Integration with mlx_distributed inference."""

from mlx_distributed import DistributedInferenceClient
from mlx_distributed.grpc_client import DistributedInferenceOrchestrator

class TrainingInferenceAdapter:
    """Adapter to use distributed inference during training."""
    
    def __init__(self, inference_config_path: str = None):
        # Use Team A's infrastructure without modifying it
        self.client = DistributedInferenceClient(inference_config_path)
    
    def generate_for_training(self, prompts, **kwargs):
        """Generate completions for training purposes."""
        return self.client.generate_batch(prompts, **kwargs)
```

### Step 5: Independent Configuration (30 minutes)

Create Team B's own config system:

```python
# mlx_distributed_training/src/mlx_distributed_training/configs/training_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training."""
    # Training specific
    model_name: str
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 3
    
    # Distributed settings
    world_size: int = 1
    backend: str = "grpc"
    
    # Integration with inference
    use_distributed_inference: bool = False
    inference_config_path: Optional[str] = None
    
    # DON'T duplicate Team A's config!
```

### Step 6: Fix Import Paths (1 hour)

Update all imports in moved files:

```python
# Before (in mlx_distributed)
from mlx_distributed.training.optimizers import DistributedAdamW
from mlx_distributed.api.server import app

# After (in mlx_distributed_training)
from mlx_distributed_training.training.optimizers import DistributedAdamW
from mlx_distributed_training.api.server import app

# For integration with Team A's work:
from mlx_distributed.grpc_client import DistributedInferenceClient
```

### Step 7: Create Clean API Server (1 hour)

Instead of modifying Team A's distributed_api.py:

```python
# mlx_distributed_training/src/mlx_distributed_training/api/server.py
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="MLX Training API")

@app.post("/train/start")
async def start_training(config: dict):
    """Start distributed training job."""
    # Your training logic here
    pass

@app.get("/train/status")
async def training_status():
    """Get training job status."""
    pass

def main():
    # Run on different port than Team A!
    uvicorn.run(app, host="0.0.0.0", port=8200)  # Not 8100!
```

### Step 8: Update Launch Scripts (30 minutes)

Create Team B specific launch scripts:

```bash
#!/bin/bash
# mlx_distributed_training/scripts/launch_training.sh

# Don't interfere with Team A's ports or services
export TRAINING_API_PORT=8200
export TENSORBOARD_PORT=6006

# Launch training API
uv run mlx-train-serve &

# Launch TensorBoard
uv run tensorboard --logdir=./logs --port=$TENSORBOARD_PORT &
```

## Benefits of Migration

### For Team A:
- ✅ No more file conflicts
- ✅ Can work on infrastructure without interference
- ✅ Clear ownership of distributed_api.py

### For Team B:
- ✅ Full control over training code
- ✅ Own API endpoints and configuration
- ✅ Can iterate quickly without breaking inference

### For Integration:
- ✅ Clean interfaces between projects
- ✅ Can import each other as dependencies
- ✅ No more port conflicts (8100 vs 8200)

## Migration Checklist

- [ ] Create mlx_distributed_training directory structure
- [ ] Create independent pyproject.toml
- [ ] Move all Team B files from mlx_distributed
- [ ] Revert Team A's files to original state
- [ ] Update all import paths
- [ ] Create integration adapters (not modifications)
- [ ] Test everything works in new location
- [ ] Update documentation

## Timeline

**Today (4 hours)**:
1. Create directory and move files (1 hour)
2. Fix imports and dependencies (1 hour)
3. Create integration layer (1 hour)
4. Test basic functionality (1 hour)

**Tomorrow**:
1. Complete API server migration
2. Update all examples and tests
3. Document the new structure

## Critical Rule

**NEVER MODIFY TEAM A's FILES!**
- Don't touch distributed_api.py
- Don't modify distributed_config.py
- Don't change grpc_server.py or grpc_client.py
- Create adapters instead!

This migration will eliminate conflicts and let both teams work independently!