# MLX Unified Training Platform

A comprehensive, unified training platform that combines Knowledge Distillation, RLHF, and Core Training capabilities into a single, cohesive framework.

## Overview

This platform provides an integrated solution for advanced model training workflows, combining:
- **Knowledge Distillation (KD)**: Multi-teacher distillation with adaptive learning
- **RLHF**: Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO)
- **Core Training**: Supervised Fine-Tuning (SFT) with LoRA/QLoRA support

## Architecture

```
mlx_unified_training/
├── src/
│   ├── core/              # Core training infrastructure
│   ├── distillation/      # Knowledge distillation module
│   ├── rlhf/              # RLHF implementations
│   ├── optimizers/        # Unified optimizers
│   ├── datasets/          # Dataset management
│   ├── api/               # FastAPI server
│   └── workflows/         # Training workflow orchestration
├── configs/               # Configuration files
├── examples/              # Example scripts
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Features

### Unified Training Pipeline
- **Sequential Training**: SFT → Distillation → RLHF
- **Parallel Training**: Run multiple training types simultaneously
- **Hybrid Training**: Combine techniques in custom workflows

### Knowledge Distillation
- Multi-teacher distillation
- Adaptive temperature scaling
- Feature matching and attention transfer
- Progressive distillation strategies

### RLHF Capabilities
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- Reward model training
- Preference dataset management

### Core Training
- Supervised Fine-Tuning (SFT)
- LoRA and QLoRA support
- Distributed training with multiple backends
- Advanced optimizers (AdamW, Lion, SGD)

### Workflow Management
- Automatic pipeline orchestration
- Checkpoint management across stages
- Resource optimization
- Progress tracking and monitoring

## Quick Start

```python
from mlx_unified import UnifiedTrainer, TrainingPipeline

# Create a unified training pipeline
pipeline = TrainingPipeline(
    stages=["sft", "distill", "rlhf"],
    model="mlx-community/Qwen2.5-1.5B",
    dataset="path/to/dataset"
)

# Configure each stage
pipeline.configure_sft(
    use_lora=True,
    lora_rank=16,
    epochs=3
)

pipeline.configure_distillation(
    teacher_models=["gpt-4", "claude-3"],
    temperature=3.0,
    alpha=0.7
)

pipeline.configure_rlhf(
    method="dpo",
    beta=0.1,
    preference_dataset="path/to/preferences"
)

# Run the unified pipeline
results = pipeline.run()
```

## API Endpoints

The platform runs on port 8600 and provides:

- `/v1/pipelines/create` - Create a new training pipeline
- `/v1/pipelines/{id}/run` - Execute a pipeline
- `/v1/pipelines/{id}/status` - Get pipeline status
- `/v1/train/sft` - Direct SFT training
- `/v1/train/distill` - Direct distillation
- `/v1/train/rlhf` - Direct RLHF training
- `/v1/workflows/templates` - Pre-configured workflow templates

## Installation

```bash
cd mlx_unified_training
pip install -e .
```

## Running the Server

```bash
python src/api/server.py
# Server runs on http://localhost:8600
```

## License

MIT License