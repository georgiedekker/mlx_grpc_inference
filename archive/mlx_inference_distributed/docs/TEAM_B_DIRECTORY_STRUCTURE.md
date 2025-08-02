# Team B Directory Structure Recommendation

## Proposed Structure: `mlx_distributed_training/`

Create a separate directory at the same level as `mlx_distributed`:

```
/Users/mini1/Movies/
├── mlx_distributed/          # Main distributed inference (Team A)
├── mlx_distributed_training/ # Team B's dedicated directory
└── mlx_rlhf/                # Team C's standalone package
```

## Directory Layout for Team B

```
mlx_distributed_training/
├── pyproject.toml           # Own package configuration
├── README.md               # Training-specific documentation
├── .gitignore
├── requirements.txt         # Training-specific dependencies
│
├── src/
│   └── mlx_distributed_training/
│       ├── __init__.py
│       ├── distributed_trainer.py      # Core distributed trainer
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py            # Dataset loaders
│       │   └── preprocessing.py      # Data preprocessing
│       ├── optimizers/
│       │   ├── __init__.py
│       │   ├── distributed_adamw.py   # Already created
│       │   └── distributed_sgd.py
│       ├── sft/
│       │   ├── __init__.py
│       │   ├── trainer.py            # SFT trainer
│       │   └── datasets.py           # Instruction datasets
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── checkpointing.py      # Save/load utilities
│       │   ├── metrics.py            # Training metrics
│       │   └── distributed_utils.py   # Communication helpers
│       └── configs/
│           ├── __init__.py
│           └── training_config.py    # Configuration classes
│
├── examples/
│   ├── train_sft_simple.py
│   ├── train_distributed.py
│   └── configs/
│       ├── sft_config.yaml
│       └── distributed_config.yaml
│
├── tests/
│   ├── test_distributed_trainer.py
│   ├── test_optimizers.py
│   ├── test_data_loading.py
│   └── test_checkpointing.py
│
└── scripts/
    ├── launch_training.sh
    └── benchmark_training.py
```

## Files to Copy/Move from mlx_distributed

### 1. Training Components (Move)
```bash
# From src/mlx_distributed/training/
- optimizers/distributed_adamw.py
- optimizers/test_distributed_optimizers.py
- optimizers/example_usage.py
- advanced_features.py
- recovery.py
```

### 2. Shared Dependencies (Copy & Adapt)
```python
# Create simplified versions in mlx_distributed_training/src/utils/
- distributed_utils.py  # Extract only training-relevant parts from distributed_comm.py
- model_utils.py       # Simplified version of model_abstraction.py
- device_utils.py      # Simplified version of device_capabilities.py
```

### 3. Configuration (Create New)
```python
# mlx_distributed_training/src/configs/training_config.py
@dataclass
class TrainingConfig:
    # Model
    model_name: str
    model_path: str
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    
    # Distributed
    world_size: int = 1
    backend: str = "grpc"
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 100
    
    # Checkpointing
    save_steps: int = 100
    output_dir: str = "./outputs"
```

## Integration Points with mlx_distributed

### 1. Import Strategy
```python
# Use mlx_distributed as a dependency
from mlx_distributed.grpc_client import DistributedInferenceClient
from mlx_distributed.sharding_strategy import ShardingStrategy

# Or create lightweight interfaces
from mlx_distributed_training.utils.distributed_utils import get_world_size, all_reduce
```

### 2. Shared Resources
- Use same gRPC ports (50051+) for training communication
- Share device capability detection logic
- Compatible configuration formats

### 3. Clean Separation
- No direct file dependencies
- Communication only through APIs
- Independent testing and deployment

## Benefits of This Structure

1. **Independence**: Team B can work without affecting Team A
2. **Clarity**: Clear ownership and boundaries
3. **Testing**: Isolated test suites
4. **Deployment**: Can be packaged separately
5. **Integration**: Clean API-based integration points

## Migration Steps

1. Create directory structure
2. Move Team B's files
3. Create standalone pyproject.toml
4. Update imports in moved files
5. Create integration tests
6. Document API contracts

This structure follows the same successful pattern as Team C's mlx_rlhf!