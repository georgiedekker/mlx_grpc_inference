"""
LoRA API Layer - MLX-free version for Team B's API endpoints
This provides LoRA configuration and utilities without requiring MLX installation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class LoRAConfigAPI:
    """LoRA configuration for API responses - no MLX dependency."""
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    use_qlora: bool = False
    
    def calculate_benefits(self, base_params: int = 1500000000) -> Dict[str, Any]:
        """Calculate LoRA benefits without MLX."""
        if self.r <= 0:
            return {
                "memory_savings_pct": 0,
                "speed_improvement": "1x",
                "trainable_params": base_params,
                "trainable_params_pct": 100.0
            }
        
        # Estimate trainable parameters for typical transformer
        # Approximation: rank * 2 * hidden_dim for each target module
        hidden_dim = 4096  # Typical for 7B models
        num_modules = len(self.target_modules)
        trainable_params = self.r * 2 * hidden_dim * num_modules
        
        return {
            "memory_savings_pct": max(0, min(95, 90 - (self.r - 8) * 2)),
            "speed_improvement": f"{min(6, 2 + self.r // 8)}x",
            "trainable_params": trainable_params,
            "trainable_params_pct": (trainable_params / base_params) * 100,
            "checkpoint_size_mb": max(0.5, trainable_params * 2 / (1024 * 1024))  # FP16
        }
    
    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        benefits = self.calculate_benefits()
        return {
            "enabled": True,
            "rank": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "use_qlora": self.use_qlora,
            **benefits
        }


def validate_lora_config(config_dict: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate LoRA configuration without MLX."""
    errors = []
    
    # Check rank
    r = config_dict.get("lora_r", config_dict.get("r", 8))
    if not isinstance(r, int) or r <= 0:
        errors.append("LoRA rank must be a positive integer")
    elif r > 128:
        errors.append("LoRA rank should typically be <= 128 for efficiency")
    
    # Check alpha
    alpha = config_dict.get("lora_alpha", config_dict.get("alpha", 16.0))
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        errors.append("LoRA alpha must be a positive number")
    
    # Check dropout
    dropout = config_dict.get("lora_dropout", config_dict.get("dropout", 0.05))
    if not isinstance(dropout, (int, float)) or not 0 <= dropout <= 1:
        errors.append("LoRA dropout must be between 0 and 1")
    
    return len(errors) == 0, errors


def get_recommended_lora_config(model_size: str) -> LoRAConfigAPI:
    """Get recommended LoRA configuration based on model size."""
    recommendations = {
        "small": LoRAConfigAPI(r=8, alpha=16.0, dropout=0.05),    # < 3B params
        "medium": LoRAConfigAPI(r=16, alpha=32.0, dropout=0.1),   # 3B - 13B params  
        "large": LoRAConfigAPI(r=32, alpha=64.0, dropout=0.1),    # > 13B params
        "default": LoRAConfigAPI(r=16, alpha=32.0, dropout=0.1)
    }
    
    # Extract size from model name
    model_lower = model_size.lower()
    if any(size in model_lower for size in ["1.5b", "1b", "0.5b"]):
        return recommendations["small"]
    elif any(size in model_lower for size in ["7b", "8b", "3b"]):
        return recommendations["medium"]
    elif any(size in model_lower for size in ["13b", "30b", "65b", "70b"]):
        return recommendations["large"]
    else:
        return recommendations["default"]


# Export commonly used functions
__all__ = [
    "LoRAConfigAPI",
    "validate_lora_config", 
    "get_recommended_lora_config"
]