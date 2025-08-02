"""
LoRA Integration for Team B's Training API

This module provides the integration points for adding LoRA/QLoRA support
to the existing training API running on port 8200.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Import the archived LoRA implementation
# Note: Team B should copy the archived lora.py to their src directory
# from mlx_distributed_training.training.lora.lora import (
#     LoRAConfig, LoRALayer, LoRALinear, 
#     apply_lora_to_model, freeze_base_model,
#     get_lora_parameters, save_lora_weights,
#     load_lora_weights, merge_lora_weights,
#     print_lora_info
# )

@dataclass
class LoRATrainingConfig:
    """Extended configuration for LoRA training in the API."""
    # LoRA parameters
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    lora_modules_to_save: List[str] = None
    lora_bias: str = "none"  # "none", "all", "lora_only"
    
    # QLoRA parameters
    use_qlora: bool = False
    qlora_compute_dtype: str = "float16"
    qlora_use_double_quant: bool = True
    qlora_quant_type: str = "nf4"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                                       "gate_proj", "up_proj", "down_proj"]
        if self.lora_modules_to_save is None:
            self.lora_modules_to_save = []
    
    def to_lora_config(self):
        """Convert to LoRAConfig for the archived implementation."""
        return {
            "r": self.lora_r,
            "alpha": self.lora_alpha,
            "dropout": self.lora_dropout,
            "target_modules": self.lora_target_modules,
            "modules_to_save": self.lora_modules_to_save,
            "bias": self.lora_bias,
            "task_type": "CAUSAL_LM",
            "use_qlora": self.use_qlora,
            "bnb_4bit_compute_dtype": self.qlora_compute_dtype,
            "bnb_4bit_use_double_quant": self.qlora_use_double_quant,
            "bnb_4bit_quant_type": self.qlora_quant_type
        }


def add_lora_to_training_job_request(request_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add LoRA parameters to the training job request schema.
    
    This should be integrated into the FastAPI request model.
    """
    lora_params = {
        "use_lora": {
            "type": "boolean",
            "default": False,
            "description": "Enable LoRA fine-tuning"
        },
        "lora_r": {
            "type": "integer",
            "default": 16,
            "description": "LoRA rank (r parameter)"
        },
        "lora_alpha": {
            "type": "number",
            "default": 32.0,
            "description": "LoRA scaling parameter (alpha)"
        },
        "lora_dropout": {
            "type": "number",
            "default": 0.1,
            "description": "LoRA dropout rate"
        },
        "lora_target_modules": {
            "type": "array",
            "items": {"type": "string"},
            "default": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "description": "Modules to apply LoRA to"
        },
        "use_qlora": {
            "type": "boolean",
            "default": False,
            "description": "Enable QLoRA (4-bit quantization)"
        }
    }
    
    # Add to hyperparameters section
    if "hyperparameters" in request_schema.get("properties", {}):
        request_schema["properties"]["hyperparameters"]["properties"].update(lora_params)
    else:
        # Create hyperparameters section if it doesn't exist
        request_schema["properties"]["hyperparameters"] = {
            "type": "object",
            "properties": lora_params
        }
    
    return request_schema


def create_lora_enabled_trainer(
    model: nn.Module,
    training_config: Dict[str, Any],
    tokenizer: Any = None
) -> tuple[nn.Module, Dict[str, Any]]:
    """
    Create a LoRA-enabled trainer by applying LoRA to the model.
    
    Args:
        model: The base model to apply LoRA to
        training_config: Training configuration from the API request
        tokenizer: Tokenizer (optional)
    
    Returns:
        Tuple of (lora_model, training_info)
    """
    # Extract LoRA configuration
    lora_config = LoRATrainingConfig(
        use_lora=training_config.get("use_lora", False),
        lora_r=training_config.get("lora_r", 16),
        lora_alpha=training_config.get("lora_alpha", 32.0),
        lora_dropout=training_config.get("lora_dropout", 0.1),
        lora_target_modules=training_config.get("lora_target_modules"),
        use_qlora=training_config.get("use_qlora", False)
    )
    
    if not lora_config.use_lora:
        logger.info("LoRA not enabled, returning original model")
        return model, {"lora_enabled": False}
    
    logger.info(f"Applying LoRA with config: {lora_config}")
    
    # Import and use the archived LoRA implementation
    # Note: This is a placeholder - Team B needs to integrate the actual lora.py
    try:
        # Apply LoRA to model
        # lora_model = apply_lora_to_model(model, lora_config.to_lora_config())
        
        # For now, return a mock response
        training_info = {
            "lora_enabled": True,
            "lora_config": asdict(lora_config),
            "trainable_parameters": "TBD - integrate lora.py",
            "total_parameters": "TBD - integrate lora.py",
            "message": "LoRA integration pending - copy lora.py from archived_components"
        }
        
        return model, training_info
        
    except Exception as e:
        logger.error(f"Failed to apply LoRA: {e}")
        raise


def update_training_metrics_with_lora(
    metrics: Dict[str, Any],
    lora_enabled: bool,
    lora_config: Optional[LoRATrainingConfig] = None
) -> Dict[str, Any]:
    """
    Update training metrics to include LoRA-specific information.
    """
    if lora_enabled and lora_config:
        metrics["lora_info"] = {
            "enabled": True,
            "rank": lora_config.lora_r,
            "alpha": lora_config.lora_alpha,
            "target_modules": lora_config.lora_target_modules,
            "qlora": lora_config.use_qlora
        }
    else:
        metrics["lora_info"] = {"enabled": False}
    
    return metrics


def save_lora_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    include_optimizer: bool = False,
    optimizer_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Save LoRA weights and training state.
    """
    try:
        # Save LoRA weights only (much smaller than full model)
        # save_lora_weights(model, checkpoint_path)
        
        # For now, return mock response
        checkpoint_info = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_type": "lora_weights",
            "size_mb": "TBD",
            "message": "LoRA checkpoint saving pending - integrate lora.py"
        }
        
        if include_optimizer and optimizer_state:
            # Save optimizer state separately
            optimizer_path = checkpoint_path.replace(".safetensors", "_optimizer.npz")
            # mx.savez(optimizer_path, **optimizer_state)
            checkpoint_info["optimizer_path"] = optimizer_path
        
        return checkpoint_info
        
    except Exception as e:
        logger.error(f"Failed to save LoRA checkpoint: {e}")
        raise


def load_lora_for_inference(
    base_model_path: str,
    lora_weights_path: str,
    merge_weights: bool = False
) -> nn.Module:
    """
    Load a model with LoRA weights for inference.
    
    Args:
        base_model_path: Path to the base model
        lora_weights_path: Path to the LoRA weights
        merge_weights: Whether to merge LoRA weights into base model
    
    Returns:
        Model ready for inference
    """
    try:
        # Load base model
        # model = load_model(base_model_path)
        
        # Load LoRA weights
        # load_lora_weights(model, lora_weights_path)
        
        # Optionally merge weights for faster inference
        # if merge_weights:
        #     model = merge_lora_weights(model)
        
        # For now, return None with message
        logger.info(f"LoRA inference loading pending - integrate lora.py")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"LoRA weights: {lora_weights_path}")
        logger.info(f"Merge weights: {merge_weights}")
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to load LoRA model: {e}")
        raise


# API endpoint integration examples
def example_training_job_with_lora():
    """
    Example of how to create a training job with LoRA enabled.
    """
    training_request = {
        "model": "mlx-community/Qwen2.5-1.5B-4bit",
        "training_file": "alpaca_data.json",
        "hyperparameters": {
            "n_epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-5,
            # LoRA parameters
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16.0,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            # QLoRA for 4-bit training
            "use_qlora": True,
            "qlora_compute_dtype": "float16"
        }
    }
    
    return training_request


def integrate_lora_into_api():
    """
    Instructions for integrating LoRA into Team B's API.
    """
    integration_steps = """
    # LoRA Integration Steps for Team B
    
    1. Copy the archived LoRA implementation:
       ```bash
       cp mlx_knowledge_distillation/mlx_distributed_training/archived_components/lora/lora.py \\
          /Users/mini1/Movies/mlx_distributed_training/src/mlx_distributed_training/training/lora/
       ```
    
    2. Update your FastAPI training endpoint to accept LoRA parameters:
       ```python
       class TrainingJobRequest(BaseModel):
           model: str
           training_file: str
           hyperparameters: Dict[str, Any] = Field(default_factory=dict)
           
       # In hyperparameters, accept:
       # - use_lora: bool
       # - lora_r: int
       # - lora_alpha: float
       # - lora_target_modules: List[str]
       # - use_qlora: bool
       ```
    
    3. In your training logic, apply LoRA before training:
       ```python
       from mlx_distributed_training.training.lora import apply_lora_to_model, LoRAConfig
       
       if hyperparameters.get("use_lora", False):
           lora_config = LoRAConfig(
               r=hyperparameters.get("lora_r", 16),
               alpha=hyperparameters.get("lora_alpha", 32.0),
               dropout=hyperparameters.get("lora_dropout", 0.1),
               target_modules=hyperparameters.get("lora_target_modules"),
               use_qlora=hyperparameters.get("use_qlora", False)
           )
           model = apply_lora_to_model(model, lora_config)
       ```
    
    4. Update checkpoint saving to use LoRA-specific saving:
       ```python
       from mlx_distributed_training.training.lora import save_lora_weights
       
       if lora_enabled:
           save_lora_weights(model, checkpoint_path)
       else:
           # Regular model saving
       ```
    
    5. Add LoRA info to training status/metrics:
       ```python
       if lora_enabled:
           status["lora_info"] = {
               "rank": lora_config.r,
               "alpha": lora_config.alpha,
               "trainable_params": count_trainable_params(model)
           }
       ```
    """
    
    return integration_steps


if __name__ == "__main__":
    # Print integration instructions
    print(integrate_lora_into_api())
    
    # Show example request
    print("\nExample LoRA training request:")
    import json
    print(json.dumps(example_training_job_with_lora(), indent=2))