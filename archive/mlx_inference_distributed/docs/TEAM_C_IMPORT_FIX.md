# Team C Import Fix Guide

## Issue: MLX Optimizer Import Errors

Team C is using incorrect import paths for MLX optimizers. Here's the fix:

### Current (Wrong):
```python
# In src/mlx_rlhf/training/dpo.py:95
self.optimizer = mx.optimizers.AdamW(...)

# In src/mlx_rlhf/training/ppo.py:106  
self.policy_optimizer = mx.optimizers.AdamW(...)
```

### Fix Required:
```python
# Add this import at the top of files
import mlx.optimizers as optim

# Then use:
self.optimizer = optim.AdamW(
    learning_rate=config.learning_rate,
    betas=(config.adam_beta1, config.adam_beta2),
    eps=config.adam_epsilon,
    weight_decay=config.weight_decay
)
```

### Files to Fix:
1. `src/mlx_rlhf/training/dpo.py` - Line 95
2. `src/mlx_rlhf/training/ppo.py` - Line 106 and similar
3. `src/mlx_rlhf/training/reward_model.py` - Check for similar usage
4. Any other training files using optimizers

### Other MLX Import Patterns:
```python
# Core MLX
import mlx.core as mx

# Optimizers
import mlx.optimizers as optim

# Neural network layers
import mlx.nn as nn

# Learning rate schedulers
from mlx.optimizers import cosine_decay, step_decay
```

## Good News: Setup Issues Fixed! ✅

- Python version: Now correctly set to >=3.13
- MLX versions: Updated to mlx>=0.27.1, mlx-lm>=0.26.2
- Virtual environment: Created at mlx_rlhf/mlx_rlhf/.venv/

## Next Steps:

1. Fix all optimizer imports
2. Run tests to verify fixes
3. Generate uv.lock file:
   ```bash
   cd /Users/mini1/Movies/mlx_distributed/mlx_rlhf
   uv pip install -e .
   # This will create uv.lock
   ```

Keep up the good work Team C!