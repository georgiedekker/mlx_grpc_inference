"""
Core LoRA (Low-Rank Adaptation) Implementation for MLX
Production-ready LoRA implementation for parameter-efficient fine-tuning.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    
    r: int = 16  # Rank of adaptation
    alpha: float = 32.0  # LoRA scaling parameter
    dropout: float = 0.1  # LoRA dropout
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"  # Bias type: "none", "all", or "lora_only"
    use_qlora: bool = False  # Use QLoRA (4-bit quantization)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"LoRA dropout must be in [0, 1], got {self.dropout}")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"LoRA bias must be 'none', 'all', or 'lora_only', got {self.bias}")


class LoRALayer(nn.Module):
    """Core LoRA layer implementation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices A and B
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize LoRA weights following standard practice."""
        # Initialize A with random normal, B with zeros
        nn.init.normal(self.lora_A.weight, std=1/self.r)
        nn.init.zeros(self.lora_B.weight)
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through LoRA layer."""
        # x @ A @ B * scaling
        result = self.lora_A(x)
        if self.dropout is not None:
            result = self.dropout(result)
        result = self.lora_B(result)
        return result * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.lora = LoRALayer(
            in_features=base_layer.weight.shape[1],
            out_features=base_layer.weight.shape[0],
            r=r,
            alpha=alpha,
            dropout=dropout
        )
        
        # Track if LoRA is merged into base weights
        self.merged = False
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass combining base layer and LoRA adaptation."""
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into base layer for deployment."""
        if not self.merged:
            # Compute LoRA weight: A @ B * scaling
            lora_weight = self.lora.lora_B.weight @ self.lora.lora_A.weight * self.lora.scaling
            # Add to base weight
            self.base_layer.weight = self.base_layer.weight + lora_weight
            self.merged = True
    
    def unmerge_weights(self):
        """Separate LoRA weights from base layer."""
        if self.merged:
            # Subtract LoRA weight from base
            lora_weight = self.lora.lora_B.weight @ self.lora.lora_A.weight * self.lora.scaling
            self.base_layer.weight = self.base_layer.weight - lora_weight
            self.merged = False


def apply_lora_to_model(
    model: nn.Module, 
    config: LoRAConfig,
    verbose: bool = True
) -> nn.Module:
    """Apply LoRA to specified modules in a model."""
    
    if verbose:
        print(f"🔧 Applying LoRA with rank={config.r}, alpha={config.alpha}")
        print(f"   Target modules: {config.target_modules}")
    
    lora_modules_count = 0
    
    def apply_lora_to_layer(module, name=""):
        """Recursively apply LoRA to target modules."""
        nonlocal lora_modules_count
        
        # Check if this module should be replaced with LoRA
        module_name = name.split(".")[-1] if "." in name else name
        
        if isinstance(module, nn.Linear) and module_name in config.target_modules:
            # Replace with LoRA version
            lora_module = LoRALinear(
                base_layer=module,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout
            )
            lora_modules_count += 1
            if verbose:
                print(f"   ✅ Applied LoRA to {name} ({module.weight.shape})")
            return lora_module
        
        # Recursively process child modules
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            new_child = apply_lora_to_layer(child_module, full_name)
            if new_child is not child_module:
                setattr(module, child_name, new_child)
        
        return module
    
    # Apply LoRA to the model
    model = apply_lora_to_layer(model)
    
    if verbose:
        print(f"✅ Applied LoRA to {lora_modules_count} modules")
    
    if lora_modules_count == 0:
        print("⚠️  Warning: No modules were modified. Check target_modules names.")
    
    return model


def freeze_base_model(model: nn.Module):
    """Freeze base model parameters, keeping only LoRA parameters trainable."""
    frozen_params = 0
    trainable_params = 0
    
    def freeze_non_lora_params(module, name=""):
        nonlocal frozen_params, trainable_params
        
        if isinstance(module, LoRALinear):
            # Freeze base layer parameters
            if hasattr(module.base_layer, 'weight'):
                module.base_layer.weight.requires_grad = False
                frozen_params += module.base_layer.weight.size
            if hasattr(module.base_layer, 'bias') and module.base_layer.bias is not None:
                module.base_layer.bias.requires_grad = False
                frozen_params += module.base_layer.bias.size
            
            # Keep LoRA parameters trainable
            for param in module.lora.parameters():
                param.requires_grad = True
                trainable_params += param.size
        else:
            # For non-LoRA modules, freeze all parameters
            for param in module.parameters():
                if hasattr(param, 'requires_grad'):
                    param.requires_grad = False
                    frozen_params += param.size
        
        # Recursively process children
        for child in module.children():
            freeze_non_lora_params(child)
    
    freeze_non_lora_params(model)
    
    print(f"🧊 Frozen {frozen_params:,} base parameters")
    print(f"🔥 Keeping {trainable_params:,} LoRA parameters trainable")
    print(f"📊 Parameter efficiency: {trainable_params/(frozen_params + trainable_params)*100:.2f}% trainable")


def get_lora_parameters(model: nn.Module) -> Dict[str, mx.array]:
    """Extract only LoRA parameters from the model."""
    lora_params = {}
    
    def extract_lora_params(module, prefix=""):
        if isinstance(module, LoRALinear):
            # Extract LoRA A and B matrices
            lora_prefix = f"{prefix}.lora" if prefix else "lora"
            for name, param in module.lora.named_parameters():
                param_name = f"{lora_prefix}.{name}"
                lora_params[param_name] = param
        
        # Recursively process children
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            extract_lora_params(child, child_prefix)
    
    extract_lora_params(model)
    return lora_params


def save_lora_weights(
    model: nn.Module, 
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save only LoRA weights to file."""
    lora_params = get_lora_parameters(model)
    
    if not lora_params:
        raise ValueError("No LoRA parameters found in model")
    
    # Prepare save data
    save_data = {"lora_weights": lora_params}
    if metadata:
        save_data["metadata"] = metadata
    
    # Save using MLX
    mx.save_safetensors(file_path, save_data)
    
    total_params = sum(p.size for p in lora_params.values())
    print(f"💾 Saved {len(lora_params)} LoRA parameters ({total_params:,} total) to {file_path}")


def load_lora_weights(
    model: nn.Module, 
    file_path: str
) -> Dict[str, Any]:
    """Load LoRA weights from file."""
    # Load data
    data = mx.load(file_path)
    
    if "lora_weights" not in data:
        raise ValueError("File does not contain LoRA weights")
    
    lora_weights = data["lora_weights"]
    
    # Apply weights to model
    def apply_lora_weights(module, prefix=""):
        if isinstance(module, LoRALinear):
            lora_prefix = f"{prefix}.lora" if prefix else "lora"
            
            # Load A and B matrices
            for name, param in module.lora.named_parameters():
                param_name = f"{lora_prefix}.{name}"
                if param_name in lora_weights:
                    param.update(lora_weights[param_name])
        
        # Recursively process children
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            apply_lora_weights(child, child_prefix)
    
    apply_lora_weights(model)
    
    metadata = data.get("metadata", {})
    print(f"📥 Loaded LoRA weights from {file_path}")
    
    return metadata


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into base model weights."""
    merged_count = 0
    
    def merge_module_lora(module):
        nonlocal merged_count
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merged_count += 1
        
        for child in module.children():
            merge_module_lora(child)
    
    merge_module_lora(model)
    print(f"🔀 Merged {merged_count} LoRA modules into base weights")


def print_lora_info(model: nn.Module, config: LoRAConfig):
    """Print detailed LoRA configuration and parameter information."""
    print("\n" + "="*60)
    print("🔧 LoRA Configuration Summary")
    print("="*60)
    print(f"Rank (r): {config.r}")
    print(f"Alpha: {config.alpha}")
    print(f"Scaling: {config.alpha / config.r:.2f}")
    print(f"Dropout: {config.dropout}")
    print(f"Target modules: {', '.join(config.target_modules)}")
    print(f"QLoRA enabled: {config.use_qlora}")
    
    # Count parameters
    total_params = sum(p.size for p in model.parameters())
    lora_params = get_lora_parameters(model)
    trainable_params = sum(p.size for p in lora_params.values())
    
    print(f"\n📊 Parameter Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.3f}%")
    print(f"Memory savings: {(1 - trainable_params/total_params)*100:.1f}%")
    print(f"Compression ratio: {total_params/trainable_params:.1f}x")
    print("="*60)


# Utility functions for model preparation
def prepare_model_for_lora_training(
    model: nn.Module,
    config: LoRAConfig,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Complete model preparation for LoRA training."""
    
    if verbose:
        print("🚀 Preparing model for LoRA training...")
    
    # Apply LoRA
    model = apply_lora_to_model(model, config, verbose=verbose)
    
    # Freeze base parameters
    freeze_base_model(model)
    
    # Get statistics
    total_params = sum(p.size for p in model.parameters())
    lora_params = get_lora_parameters(model)
    trainable_params = sum(p.size for p in lora_params.values())
    
    info = {
        "lora_config": config,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params,
        "memory_savings_pct": (1 - trainable_params / total_params) * 100,
        "compression_ratio": total_params / trainable_params if trainable_params > 0 else 1.0
    }
    
    if verbose:
        print_lora_info(model, config)
    
    return model, info


# Example usage and configuration presets
LORA_PRESETS = {
    "memory_efficient": LoRAConfig(r=4, alpha=8.0, dropout=0.05),
    "balanced": LoRAConfig(r=8, alpha=16.0, dropout=0.1),
    "high_capacity": LoRAConfig(r=16, alpha=32.0, dropout=0.1),
    "research": LoRAConfig(r=32, alpha=64.0, dropout=0.1),
}

def get_lora_preset(preset_name: str) -> LoRAConfig:
    """Get a predefined LoRA configuration."""
    if preset_name not in LORA_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(LORA_PRESETS.keys())}")
    return LORA_PRESETS[preset_name]