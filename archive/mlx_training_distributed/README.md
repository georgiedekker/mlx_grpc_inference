# Training MLX

🚀 **Clean, production-ready distributed training system for MLX models**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![API Port](https://img.shields.io/badge/API-Port%208200-green.svg)](http://localhost:8200)

## Overview

Training MLX is a refactored and cleaned-up distributed training system for MLX models. It provides:

- 🎯 **LoRA/QLoRA fine-tuning** - Efficient parameter-efficient training
- 📊 **Multiple dataset formats** - Alpaca and ShareGPT support
- 🌐 **RESTful API** - OpenAI-compatible endpoints on port 8200
- 🔐 **Enterprise security** - Built-in secret scanning and protection
- 🤝 **Distributed integration** - Optional distributed inference support

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ml-explore/training-mlx.git
cd training-mlx

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Optional: Install MLX support
pip install -e ".[mlx]"

# Optional: Install full training dependencies
pip install -e ".[training]"
```

### Security Setup

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
vim .env

# Run security check
python scripts/security_check.py
```

### Basic Usage

```bash
# Generate a training configuration
training-mlx init --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

# Validate your dataset
training-mlx validate data/train.jsonl --format alpaca

# Start training
training-mlx train config.yaml --name my-experiment

# Start API server
training-mlx serve
```

## API Usage

The API runs on port 8200 and provides OpenAI-compatible endpoints:

```python
import requests

# Create a training job
response = requests.post(
    "http://localhost:8200/v1/fine-tuning/jobs",
    json={
        "experiment_name": "alpaca-lora",
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "dataset_path": "data/alpaca.json",
        "dataset_format": "alpaca",
        "use_lora": True
    }
)
job = response.json()
print(f"Created job: {job['job_id']}")

# Check job status
status = requests.get(
    f"http://localhost:8200/v1/fine-tuning/jobs/{job['job_id']}"
).json()
print(f"Status: {status['status']}")
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `training-mlx init` | Generate training configuration |
| `training-mlx train CONFIG` | Start training |
| `training-mlx validate DATA` | Validate dataset |
| `training-mlx serve` | Start API server |
| `training-mlx status` | Show system status |
| `training-mlx doctor` | Diagnose issues |

## Project Structure

```
mlx_training_distributed/
├── src/training_mlx/         # Main package
│   ├── api/                  # RESTful API
│   ├── cli/                  # Command-line interface
│   ├── training/             # Training logic
│   ├── adapters/             # Integration adapters
│   ├── utils/                # Utilities
│   └── security/             # Security features
├── scripts/                  # Helper scripts
├── examples/                 # Example datasets
├── tests/                    # Test suite
├── .env.example             # Environment template
├── .gitignore               # Security gitignore
└── pyproject.toml           # Package configuration
```

## Dataset Formats

### Alpaca Format
```json
{
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
}
```

### ShareGPT Format
```json
{
    "conversations": [
        {"from": "human", "value": "What is the capital of France?"},
        {"from": "assistant", "value": "The capital of France is Paris."}
    ]
}
```

## Configuration

Example training configuration:

```yaml
experiment:
  name: "qwen-lora-training"
  tags: ["mlx", "lora", "alpaca"]

model:
  name: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
  quantization: "4bit"

training:
  batch_size: 4
  learning_rate: 5e-5
  num_epochs: 3
  gradient_accumulation_steps: 4
  warmup_ratio: 0.1

data:
  format: "alpaca"
  train_path: "data/train.json"
  validation_split: 0.1
  max_seq_length: 2048

lora:
  enabled: true
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]
```

## Security

Training MLX includes comprehensive security features:

- 🔒 **Secret Protection**: `.gitignore` blocks API keys and secrets
- 🔍 **Security Scanner**: Pre-commit checks for hardcoded secrets
- 📋 **Safe Templates**: `.env.example` for configuration
- 🛡️ **Best Practices**: Security guidelines in documentation

Run security checks:
```bash
python scripts/security_check.py
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/

# Type checking
mypy src/
```

## Integration with Distributed Inference

Training MLX can optionally integrate with distributed inference systems:

```python
from training_mlx.adapters import get_integration_status

status = get_integration_status()
if status["distributed_available"]:
    print("✅ Can use distributed inference")
else:
    print("⚠️  Using local inference")
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run security checks
4. Submit a pull request

## Support

- 📚 Documentation: [https://training-mlx.readthedocs.io/](https://training-mlx.readthedocs.io/)
- 🐛 Issues: [GitHub Issues](https://github.com/ml-explore/training-mlx/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/ml-explore/training-mlx/discussions)

---

Built with ❤️ by George Dekker