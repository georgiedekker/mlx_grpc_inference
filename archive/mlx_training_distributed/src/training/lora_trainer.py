#!/usr/bin/env python3
"""
LoRA/QLoRA training implementation for MLX
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path
import json

class LoRALinear(nn.Module):
    """Low-Rank Adaptation for Linear layers."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        use_qora: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # QLoRA quantization
        self.use_qora = use_qora
        if use_qora:
            self._quantize_base_weights()
            
        # Initialize LoRA weights
        self._init_lora_weights()
        
    def _init_lora_weights(self):
        """Initialize LoRA weights."""
        # Initialize A with Gaussian, B with zeros
        nn.init.normal(self.lora_A.weight, std=1.0/np.sqrt(self.in_features))
        nn.init.zeros(self.lora_B.weight)
        
    def _quantize_base_weights(self):
        """Quantize base weights to 4-bit for QLoRA."""
        # Simple 4-bit quantization
        weight = self.linear.weight
        
        # Get scale and zero point
        min_val = mlx.min(weight)
        max_val = mlx.max(weight)
        scale = (max_val - min_val) / 15  # 4-bit = 16 levels
        zero_point = -min_val / scale
        
        # Quantize
        quantized = mlx.round((weight - min_val) / scale)
        quantized = mx.clip(quantized, 0, 15)
        
        # Store quantization params
        self.weight_scale = scale
        self.weight_zero_point = zero_point
        self.weight_quantized = quantized.astype(mlx.uint8)
        
        # Free original weights
        self.linear.weight = None
        
    def _dequantize_weight(self):
        """Dequantize weight for forward pass."""
        if self.use_qora and hasattr(self, 'weight_quantized'):
            return self.weight_quantized.astype(mlx.float32) * self.weight_scale + self.weight_zero_point
        return self.linear.weight
        
    def forward(self, x):
        """Forward pass with LoRA."""
        # Get base weight (dequantized if using QLoRA)
        if self.use_qora:
            base_weight = self._dequantize_weight()
            base_out = x @ base_weight.T
            if self.linear.bias is not None:
                base_out = base_out + self.linear.bias
        else:
            base_out = self.linear(x)
            
        # Apply LoRA
        if self.dropout:
            x = self.dropout(x)
            
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        
        return base_out + lora_out
        
    def merge_weights(self):
        """Merge LoRA weights into base weights."""
        if self.use_qora:
            base_weight = self._dequantize_weight()
        else:
            base_weight = self.linear.weight
            
        # Compute merged weight
        lora_weight = self.lora_B.weight @ self.lora_A.weight * self.scaling
        merged_weight = base_weight + lora_weight.T
        
        # Update linear layer
        self.linear.weight = merged_weight
        
        # Reset LoRA weights
        self._init_lora_weights()
        

class LoRAConfig:
    """Configuration for LoRA training."""
    
    def __init__(
        self,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        target_modules: List[str] = None,
        use_qora: bool = False,
        qora_bits: int = 4
    ):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.use_qora = use_qora
        self.qora_bits = qora_bits
        

def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> Dict[str, LoRALinear]:
    """Apply LoRA to target modules in the model."""
    lora_modules = {}
    
    def replace_with_lora(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear) and any(target in name for target in config.target_modules):
                # Replace with LoRA
                lora_linear = LoRALinear(
                    in_features=child.weight.shape[1],
                    out_features=child.weight.shape[0],
                    r=config.r,
                    alpha=config.alpha,
                    dropout=config.dropout,
                    use_qora=config.use_qora
                )
                
                # Copy original weights
                lora_linear.linear.weight = child.weight
                if child.bias is not None:
                    lora_linear.linear.bias = child.bias
                    
                # Freeze original parameters
                lora_linear.linear.weight.requires_grad = False
                if lora_linear.linear.bias is not None:
                    lora_linear.linear.bias.requires_grad = False
                    
                setattr(module, name, lora_linear)
                lora_modules[full_name] = lora_linear
                
                print(f"Applied LoRA to {full_name}: {child.weight.shape}")
            else:
                replace_with_lora(child, full_name)
                
    replace_with_lora(model)
    
    # Calculate trainable parameters
    total_params = sum(p.size for p in model.parameters())
    trainable_params = sum(p.size for p in model.trainable_parameters())
    
    print(f"\nLoRA Configuration:")
    print(f"  Rank: {config.r}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Dropout: {config.dropout}")
    print(f"  QLoRA: {config.use_qora}")
    print(f"  Target modules: {config.target_modules}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return lora_modules


def save_lora_weights(lora_modules: Dict[str, LoRALinear], path: str):
    """Save only LoRA weights."""
    Path(path).mkdir(parents=True, exist_ok=True)
    
    lora_state = {}
    for name, module in lora_modules.items():
        lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight
        lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight
        
    mx.savez(f"{path}/lora_weights.npz", **lora_state)
    
    # Save config
    config_data = {
        "r": lora_modules[list(lora_modules.keys())[0]].r,
        "alpha": lora_modules[list(lora_modules.keys())[0]].alpha,
        "target_modules": list(lora_modules.keys())
    }
    with open(f"{path}/lora_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
        
    print(f"Saved LoRA weights to {path}")
    

def load_lora_weights(model: nn.Module, path: str) -> Dict[str, LoRALinear]:
    """Load LoRA weights and apply to model."""
    # Load config
    with open(f"{path}/lora_config.json", "r") as f:
        config_data = json.load(f)
        
    # Create LoRA config
    config = LoRAConfig(
        r=config_data["r"],
        alpha=config_data["alpha"]
    )
    
    # Apply LoRA to model
    lora_modules = apply_lora_to_model(model, config)
    
    # Load weights
    lora_weights = mx.load(f"{path}/lora_weights.npz")
    
    for name, weight in lora_weights.items():
        module_name = name.rsplit(".", 2)[0]
        param_name = name.split(".")[-2] + "." + name.split(".")[-1]
        
        if module_name in lora_modules:
            if "lora_A.weight" in param_name:
                lora_modules[module_name].lora_A.weight = weight
            elif "lora_B.weight" in param_name:
                lora_modules[module_name].lora_B.weight = weight
                
    print(f"Loaded LoRA weights from {path}")
    return lora_modules


class LoRATrainer:
    """Specialized trainer for LoRA/QLoRA."""
    
    def __init__(self, base_trainer, lora_config: LoRAConfig):
        self.base_trainer = base_trainer
        self.lora_config = lora_config
        self.lora_modules = None
        
    def prepare_model(self):
        """Prepare model with LoRA."""
        # Load base model
        self.base_trainer.load_model()
        
        # Apply LoRA
        self.lora_modules = apply_lora_to_model(
            self.base_trainer.model,
            self.lora_config
        )
        
        # Update config
        self.base_trainer.config.use_lora = True
        
    def train(self):
        """Train with LoRA."""
        # Use base trainer's training loop
        return self.base_trainer.train()
        
    def save_lora_adapter(self, path: str):
        """Save only the LoRA adapter weights."""
        save_lora_weights(self.lora_modules, path)
        
    def merge_and_save(self, path: str):
        """Merge LoRA weights and save full model."""
        # Merge weights
        for module in self.lora_modules.values():
            module.merge_weights()
            
        # Save full model
        self.base_trainer.save_checkpoint(path)