"""Adapter to load MLX models and convert them for PyTorch distributed inference."""
import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from safetensors import safe_open
import numpy as np
from huggingface_hub import snapshot_download
import struct

logger = logging.getLogger(__name__)


class MLXModelAdapter:
    """Adapter to load MLX quantized models and make them work with PyTorch."""
    
    @staticmethod
    def get_model_path(model_name: str) -> str:
        """Get local path for MLX model, downloading if necessary."""
        if os.path.exists(model_name):
            return model_name
        
        # Download from HuggingFace if needed
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_id = model_name.replace("/", "--")
        local_path = os.path.join(cache_dir, f"models--{model_id}")
        
        if os.path.exists(local_path):
            # Find snapshot directory
            snapshots_dir = os.path.join(local_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    return os.path.join(snapshots_dir, snapshots[0])
        
        # Download if not found
        logger.info(f"Downloading model {model_name}")
        return snapshot_download(model_name)
    
    @staticmethod
    def load_mlx_weights(model_path: str) -> Dict[str, torch.Tensor]:
        """Load weights from MLX model safetensors files with proper dequantization."""
        weights = {}
        model_dir = Path(model_path)
        
        # Look for model weights file
        weight_files = list(model_dir.glob("model*.safetensors")) + list(model_dir.glob("model*.weights"))
        
        if not weight_files:
            weight_files = list(model_dir.glob("*.safetensors"))
        
        logger.info(f"Found weight files: {[f.name for f in weight_files]}")
        
        for weight_file in weight_files:
            if weight_file.suffix == ".safetensors":
                with safe_open(weight_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        weights[key] = tensor
            
        return weights
    
    @staticmethod
    def load_quantization_config(model_path: str) -> Dict[str, Any]:
        """Load MLX quantization configuration."""
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('quantization', {})
        return {}
    
    @staticmethod
    def dequantize_mlx_weight(weight: torch.Tensor, weight_name: str, quant_config: Dict) -> torch.Tensor:
        """Dequantize MLX quantized weights to float16."""
        # MLX uses different quantization formats
        # Handle both int8 and uint8 quantization
        if weight.dtype in [torch.int8, torch.uint8]:
            # MLX quantization typically includes scales and biases
            # For now, use simple dequantization
            if weight.dtype == torch.uint8:
                # Convert uint8 to float and normalize
                weight_float = weight.to(torch.float32) - 128.0
                weight_float = weight_float / 127.0
            else:
                # int8 quantization
                weight_float = weight.to(torch.float32) / 127.0
            
            return weight_float.to(torch.float16)
        
        # Already in float format
        if weight.dtype in [torch.float32, torch.float16]:
            return weight.to(torch.float16)
        
        # Unknown format - try to convert
        logger.warning(f"Unknown weight dtype {weight.dtype} for {weight_name}, converting to float16")
        return weight.to(torch.float16)
    
    @staticmethod
    def convert_mlx_to_pytorch(mlx_model_path: str) -> Tuple[Dict[str, torch.Tensor], PretrainedConfig]:
        """Convert MLX model to PyTorch-compatible format."""
        model_path = MLXModelAdapter.get_model_path(mlx_model_path)
        logger.info(f"Converting MLX model from {model_path}")
        
        # Load config
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Load quantization config
        quant_config = MLXModelAdapter.load_quantization_config(model_path)
        
        # Load MLX weights
        mlx_weights = MLXModelAdapter.load_mlx_weights(model_path)
        
        # Convert weight names and dequantize
        pytorch_weights = {}
        for name, weight in mlx_weights.items():
            # Convert MLX naming to PyTorch naming
            pytorch_name = name
            
            # Dequantize
            dequantized = MLXModelAdapter.dequantize_mlx_weight(weight, name, quant_config)
            
            pytorch_weights[pytorch_name] = dequantized
        
        logger.info(f"Converted {len(pytorch_weights)} weights")
        return pytorch_weights, config


class MLXCompatibleModel(nn.Module):
    """PyTorch model that can load MLX weights and run inference."""
    
    def __init__(self, config: PretrainedConfig, weights: Dict[str, torch.Tensor], device: torch.device = None):
        super().__init__()
        self.config = config
        self.weights = weights
        self.model_type = config.model_type
        self._device = device or torch.device('cpu')
        
        # Create the actual model based on type
        self._create_model()
        
        # Load weights
        self._load_weights()
        
        # Move to target device after loading weights
        if self._device.type != 'cpu':
            self.base_model = self.base_model.to(self._device)
    
    def _create_model(self):
        """Create the appropriate model architecture."""
        # Import here to avoid circular dependency
        from transformers import AutoModelForCausalLM
        
        # Create model directly on CPU (not meta device)
        self.base_model = AutoModelForCausalLM.from_config(self.config, torch_dtype=torch.float16)
    
    def _load_weights(self):
        """Load the converted MLX weights into the model."""
        # Get the model's state dict
        model_state_dict = self.base_model.state_dict()
        
        # Map MLX weights to PyTorch model
        loaded_keys = set()
        missing_keys = set()
        
        for pytorch_key in model_state_dict.keys():
            # Try different naming conventions
            mlx_key_candidates = [
                pytorch_key,
                pytorch_key.replace("model.", ""),
                f"model.{pytorch_key}",
                pytorch_key.replace("transformer.", "model."),
                pytorch_key.replace("lm_head.", "lm_head."),
            ]
            
            loaded = False
            for mlx_key in mlx_key_candidates:
                if mlx_key in self.weights:
                    # Check shape compatibility
                    expected_shape = model_state_dict[pytorch_key].shape
                    actual_shape = self.weights[mlx_key].shape
                    
                    if expected_shape == actual_shape:
                        model_state_dict[pytorch_key] = self.weights[mlx_key]
                        loaded_keys.add(pytorch_key)
                        loaded = True
                        break
                    else:
                        logger.warning(f"Shape mismatch for {pytorch_key}: expected {expected_shape}, got {actual_shape}")
            
            if not loaded:
                missing_keys.add(pytorch_key)
        
        logger.info(f"Loaded {len(loaded_keys)} weights, missing {len(missing_keys)}")
        if missing_keys:
            logger.debug(f"Missing keys: {list(missing_keys)[:10]}...")  # Show first 10
        
        # Load the state dict
        self.base_model.load_state_dict(model_state_dict, strict=False)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass using the base model."""
        return self.base_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
    
    def generate(self, *args, **kwargs):
        """Generate method for compatibility."""
        return self.base_model.generate(*args, **kwargs)
    
    @property
    def device(self):
        """Get model device."""
        return next(self.parameters()).device
    
    def to(self, device):
        """Move model to device."""
        self.base_model = self.base_model.to(device)
        return super().to(device)




def load_mlx_model_for_pytorch(model_name: str, device: torch.device) -> Tuple[nn.Module, AutoTokenizer]:
    """Load an MLX model and convert it for PyTorch usage."""
    logger.info(f"Loading MLX model {model_name} for PyTorch")
    
    # Check if it's an MLX model
    if "mlx-community" in model_name:
        # Convert MLX model
        weights, config = MLXModelAdapter.convert_mlx_to_pytorch(model_name)
        model = MLXCompatibleModel(config, weights, device=device)
        model.eval()
    else:
        # Load regular PyTorch model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device}
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer