# LoRA package initialization

# Try importing MLX-based LoRA if available
try:
    from .lora import LoRAConfig, LoRALayer, LoRALinear, apply_lora_to_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Always import API-only components (no MLX dependency)
from .lora_api import LoRAConfigAPI, validate_lora_config, get_recommended_lora_config

__all__ = [
    'LoRAConfigAPI', 
    'validate_lora_config', 
    'get_recommended_lora_config',
    'MLX_AVAILABLE'
]

# Add MLX components if available
if MLX_AVAILABLE:
    __all__.extend(['LoRAConfig', 'LoRALayer', 'LoRALinear', 'apply_lora_to_model'])