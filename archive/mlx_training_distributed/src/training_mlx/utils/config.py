"""
Configuration utilities for Training MLX
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import jsonschema


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    config = _expand_env_vars(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration against schema."""
    schema = {
        "type": "object",
        "required": ["model", "training", "data"],
        "properties": {
            "experiment": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                }
            },
            "model": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "quantization": {"type": ["string", "null"]},
                }
            },
            "training": {
                "type": "object",
                "required": ["batch_size", "learning_rate", "num_epochs"],
                "properties": {
                    "batch_size": {"type": "integer", "minimum": 1},
                    "learning_rate": {"type": "number", "minimum": 0},
                    "num_epochs": {"type": "integer", "minimum": 1},
                    "gradient_accumulation_steps": {"type": "integer", "minimum": 1},
                    "warmup_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                    "save_steps": {"type": "integer", "minimum": 1},
                    "eval_steps": {"type": "integer", "minimum": 1},
                }
            },
            "data": {
                "type": "object",
                "required": ["format", "train_path"],
                "properties": {
                    "format": {"type": "string", "enum": ["alpaca", "sharegpt"]},
                    "train_path": {"type": "string"},
                    "validation_split": {"type": "number", "minimum": 0, "maximum": 1},
                    "max_seq_length": {"type": "integer", "minimum": 1},
                }
            },
            "lora": {
                "type": ["object", "null"],
                "properties": {
                    "enabled": {"type": "boolean"},
                    "r": {"type": "integer", "minimum": 1},
                    "alpha": {"type": "number", "minimum": 0},
                    "dropout": {"type": "number", "minimum": 0, "maximum": 1},
                    "target_modules": {"type": "array", "items": {"type": "string"}},
                }
            },
            "distributed": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "strategy": {"type": "string", "enum": ["data_parallel", "pipeline_parallel"]},
                }
            },
            "logging": {
                "type": "object",
                "properties": {
                    "level": {"type": "string"},
                    "wandb": {"type": "object"},
                    "tensorboard": {"type": "object"},
                }
            }
        }
    }
    
    try:
        jsonschema.validate(config, schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Invalid configuration: {e.message}")


def _expand_env_vars(config: Any) -> Any:
    """Recursively expand environment variables in config."""
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Expand ${VAR} or $VAR patterns
        if config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        elif config.startswith("$"):
            var_name = config[1:]
            return os.getenv(var_name, config)
        else:
            return config
    else:
        return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configurations."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    """Get default training configuration."""
    return {
        "experiment": {
            "name": "default_experiment",
            "tags": ["mlx", "training"],
        },
        "model": {
            "name": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "quantization": "4bit",
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1,
            "save_steps": 100,
            "eval_steps": 50,
        },
        "data": {
            "format": "alpaca",
            "train_path": "data/train.json",
            "validation_split": 0.1,
            "max_seq_length": 2048,
        },
        "lora": {
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "distributed": {
            "enabled": False,
            "strategy": "data_parallel",
        },
        "logging": {
            "level": "INFO",
            "wandb": {
                "enabled": False,
                "project": "training-mlx",
            },
            "tensorboard": {
                "enabled": True,
                "log_dir": "./logs",
            },
        },
    }