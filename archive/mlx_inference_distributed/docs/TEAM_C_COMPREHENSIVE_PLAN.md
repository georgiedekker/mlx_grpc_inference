# Team C: Comprehensive Plan to Achieve Original Goals

## Current Status
✅ Fixed MLX optimizer imports
✅ Fixed model output shapes and array operations  
✅ Reward model tests passing (12/12)
❌ PPO and DPO tests still have issues
❌ Missing uv.lock file
❌ Not integrated with main mlx_distributed

## Original Goals to Achieve

### Goal 1: Complete Standalone MLX RLHF Package

#### 1.1 Fix Remaining Test Issues (Priority: HIGH)
```bash
# Current issues to fix:
```

**PPO Tests:**
- `log_softmax` location: Use `mx.log_softmax` or `nn.log_softmax`
- Missing fixtures: Add to conftest.py
- Mini-batch indexing: Fix array slicing

**DPO Tests:**
- Import missing utilities
- Fix batch generation
- Ensure proper loss computation

**Action Items:**
```python
# In PPO tests - fix log_softmax
import mlx.nn as nn
log_probs = nn.log_softmax(logits, axis=-1)

# In conftest.py - add missing fixtures
@pytest.fixture
def preference_data_file(temp_dir):
    """Create test preference data file."""
    data = {
        "examples": [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence...",
                "rejected": "AI is magic..."
            }
        ]
    }
    path = Path(temp_dir) / "preferences.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path
```

#### 1.2 Generate uv.lock (Priority: HIGH)
```bash
cd /Users/mini1/Movies/mlx_distributed/mlx_rlhf
# Remove old venv if exists
rm -rf mlx_rlhf/.venv

# Create fresh UV environment
uv venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows

# Install in editable mode - this creates uv.lock
uv pip install -e ".[dev]"

# Verify lock file created
ls -la uv.lock
```

#### 1.3 Complete Test Suite (Priority: HIGH)
```bash
# Run all tests to ensure they pass
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=mlx_rlhf --cov-report=html

# Run specific test categories
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/edge_cases/ -v
```

### Goal 2: Implement Missing RLHF Components

#### 2.1 Value Model Implementation
```python
# src/mlx_rlhf/models/value_model.py
import mlx.core as mx
import mlx.nn as nn

class ValueModel(nn.Module):
    def __init__(self, config: ValueModelConfig, base_model=None):
        super().__init__()
        self.config = config
        self.base_model = base_model or self._load_base_model()
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )
    
    def __call__(self, input_ids, attention_mask=None):
        # Get hidden states from base model
        outputs = self.base_model(input_ids, mask=attention_mask)
        hidden_states = outputs  # Assuming last hidden state
        
        # Pool hidden states (e.g., last token)
        if attention_mask is not None:
            last_token_idx = mx.sum(attention_mask, axis=1) - 1
            pooled = hidden_states[mx.arange(hidden_states.shape[0]), last_token_idx]
        else:
            pooled = hidden_states[:, -1, :]
        
        # Get value
        value = self.value_head(pooled)
        return value
```

#### 2.2 Complete PPO Implementation
```python
# Key components to implement:

1. **Advantage Estimation**:
def compute_advantages(rewards, values, next_values, gamma=0.99, lam=0.95):
    deltas = rewards + gamma * next_values - values
    advantages = mx.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lam * gae
        advantages[t] = gae
    return advantages

2. **PPO Loss**:
def ppo_loss(log_probs, old_log_probs, advantages, clip_range=0.2):
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1 - clip_range, 1 + clip_range)
    loss = -mx.minimum(ratio * advantages, clipped_ratio * advantages)
    return mx.mean(loss)

3. **KL Divergence Control**:
def adaptive_kl_control(kl_div, target_kl=0.01):
    if kl_div > 1.5 * target_kl:
        # Increase KL penalty
        return 1.5
    elif kl_div < 0.5 * target_kl:
        # Decrease KL penalty
        return 0.5
    return 1.0
```

#### 2.3 Training Scripts
```python
# examples/train_dpo.py
import mlx.core as mx
from mlx_rlhf import DPOTrainer, DPOConfig, create_preference_datasets

def main():
    # Load config
    config = DPOConfig(
        model_name_or_path="mlx-community/Mistral-7B-v0.1-hf-4bit-mlx",
        learning_rate=1e-6,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        beta=0.1,
        output_dir="./dpo_output"
    )
    
    # Create datasets
    train_loader, eval_loader = create_preference_datasets(
        "path/to/preferences.json",
        tokenizer=tokenizer,
        max_length=512,
        batch_size=config.batch_size
    )
    
    # Create trainer
    trainer = DPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=eval_loader
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```

### Goal 3: Integration Strategy (After Standalone Completion)

#### 3.1 Create Integration Interface
```python
# src/mlx_rlhf/distributed/interface.py
from typing import Optional
from mlx_distributed import DistributedInferenceClient

class DistributedRLHFInterface:
    """Interface for using mlx_rlhf with mlx_distributed."""
    
    def __init__(self, distributed_client: Optional[DistributedInferenceClient] = None):
        self.distributed_client = distributed_client
        self.use_distributed = distributed_client is not None
    
    def generate(self, prompt, **kwargs):
        if self.use_distributed:
            return self.distributed_client.generate(prompt, **kwargs)
        else:
            # Local generation
            return self.local_model.generate(prompt, **kwargs)
```

#### 3.2 Documentation
```markdown
# docs/INTEGRATION.md

## Using mlx_rlhf with mlx_distributed

### Standalone Mode (Recommended)
```bash
pip install mlx-rlhf
mlx-rlhf-train-dpo --config dpo_config.yaml
```

### Distributed Mode (Advanced)
```python
from mlx_rlhf import DPOTrainer
from mlx_distributed import DistributedInferenceClient

# Connect to distributed cluster
client = DistributedInferenceClient("grpc://cluster:50051")

# Use distributed inference in training
trainer = DPOTrainer(
    config=config,
    distributed_client=client
)
```
```

### Goal 4: Production Readiness

#### 4.1 CI/CD Pipeline
```yaml
# .github/workflows/test.yml
name: Test MLX RLHF

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Run tests
        run: |
          uv venv
          uv pip install -e ".[dev]"
          uv run pytest tests/ -v --cov
```

#### 4.2 Performance Benchmarks
```python
# benchmarks/training_speed.py
import time
import mlx.core as mx

def benchmark_dpo_training():
    # Measure tokens/second
    # Memory usage
    # Multi-device scaling
    pass

def benchmark_ppo_training():
    # Measure episodes/second
    # Value function convergence
    # KL divergence stability
    pass
```

## Execution Timeline

### Week 1 (Immediate)
1. Fix remaining test issues (2 days)
2. Generate uv.lock file (30 minutes)
3. Complete test suite validation (1 day)
4. Implement Value Model (1 day)
5. Complete PPO implementation (2 days)

### Week 2
1. Create training scripts (2 days)
2. Write documentation (1 day)
3. Performance benchmarking (1 day)
4. CI/CD setup (1 day)

### Week 3
1. Integration interface design (2 days)
2. Testing with mlx_distributed (when Team A fixes gRPC)
3. Production deployment guide (1 day)

### Week 4
1. Performance optimization
2. Advanced features (mixed precision, etc.)
3. Community release preparation

## Success Metrics

1. **All tests passing** (100% success rate)
2. **uv.lock file generated** and reproducible installs
3. **Training convergence** on standard benchmarks
4. **Documentation complete** with examples
5. **Standalone package** installable via `pip install mlx-rlhf`

## Current Blockers & Solutions

| Blocker | Solution | Priority |
|---------|----------|----------|
| Test failures | Fix imports and fixtures | HIGH |
| No uv.lock | Run `uv pip install -e .` | HIGH |
| Missing Value Model | Implement from spec | MEDIUM |
| No training scripts | Create examples | MEDIUM |
| No integration | Wait for Team A or go standalone | LOW |

## Recommendations

1. **Focus on standalone first** - Don't wait for Team A's broken gRPC
2. **Get tests green** - This validates the implementation
3. **Create working examples** - Proves the system works
4. **Document everything** - Makes it usable by others
5. **Benchmark performance** - Shows MLX advantages

Team C is closest to delivering working software. Stay focused on quality and completion!