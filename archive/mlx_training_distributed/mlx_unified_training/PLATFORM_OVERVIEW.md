# MLX Unified Training Platform Overview

## Location
`/Users/mini1/Movies/mlx_unified_training/`

## Description
A completely independent, unified training platform that combines Knowledge Distillation, RLHF, and Core Training capabilities into a single cohesive framework. This is a **separate codebase** with no imports from other MLX projects.

## Key Features

### 1. Pipeline Orchestration
- Create multi-stage training pipelines
- Automatic stage sequencing (SFT → Distillation → RLHF)
- Progress tracking and monitoring
- Checkpoint management between stages

### 2. Training Modules

#### Knowledge Distillation
- Multi-teacher distillation
- Adaptive temperature scaling
- Feature matching
- Intermediate layer distillation

#### RLHF (Reinforcement Learning from Human Feedback)
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- Reward model training
- Preference dataset handling

#### Core Training
- Supervised Fine-Tuning (SFT)
- LoRA/QLoRA support
- Distributed training
- Advanced optimizers

### 3. Workflow Templates
Pre-configured pipelines for common use cases:
- **Chatbot Training**: SFT → RLHF
- **Efficient LLM**: Distillation → SFT
- **Aligned Model**: SFT → RLHF → Distillation
- **Research Pipeline**: Full pipeline with all stages

## API Endpoints

### Pipeline Management
- `POST /v1/pipelines/create` - Create new pipeline
- `GET /v1/pipelines` - List all pipelines
- `GET /v1/pipelines/{id}` - Get pipeline details
- `POST /v1/pipelines/{id}/run` - Execute pipeline
- `GET /v1/pipelines/{id}/status` - Get execution status

### Direct Training
- `POST /v1/train/sft` - Direct SFT training
- `POST /v1/train/distill` - Direct distillation
- `POST /v1/train/rlhf` - Direct RLHF training

### Workflows
- `GET /v1/workflows/templates` - List templates
- `POST /v1/workflows/from-template` - Create from template

## Quick Start

1. Navigate to directory:
   ```bash
   cd /Users/mini1/Movies/mlx_unified_training
   ```

2. Start the server:
   ```bash
   ./start_server.sh
   ```

3. Server runs on: `http://localhost:8600`
4. API docs: `http://localhost:8600/docs`

## Example Usage

```python
import requests

# Create a unified pipeline
pipeline = {
    "name": "my_aligned_model",
    "stages": ["sft", "distillation", "rlhf"],
    "base_model": "mlx-community/Qwen2.5-1.5B",
    "dataset_path": "/path/to/data.json",
    "auto_configure": True
}

response = requests.post(
    "http://localhost:8600/v1/pipelines/create",
    json=pipeline,
    headers={"X-API-Key": "mlx-unified-key"}
)

pipeline_id = response.json()["pipeline_id"]

# Run the pipeline
requests.post(
    f"http://localhost:8600/v1/pipelines/{pipeline_id}/run",
    headers={"X-API-Key": "mlx-unified-key"}
)
```

## Integration with MLX Ecosystem

While this is a separate codebase, it conceptually unifies the capabilities of:
- Knowledge Distillation (Port 8300)
- RLHF Framework (Port 8400)
- MLX Training (Port 8500)

Into a single, orchestrated platform on Port 8600.

## Development Status
- ✅ API server implementation
- ✅ Pipeline orchestration
- ✅ Workflow templates
- 🚧 Core training logic (to be implemented)
- 🚧 Integration tests
- 🚧 Documentation

## Future Enhancements
- Real training implementation (currently simulated)
- Model registry integration
- Distributed training support
- Advanced monitoring and metrics
- Web UI for pipeline management