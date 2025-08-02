"""
Model abstraction layer for supporting multiple MLX model architectures.

This module provides a unified interface for different model types (Qwen, Llama, Mistral, etc.)
and handles model-specific sharding strategies.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    architecture: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    total_params: int
    dtype: str
    quantization: Optional[str] = None
    
    def estimate_size_gb(self) -> float:
        """Estimate model size in GB based on parameters and dtype."""
        bytes_per_param = {
            'float32': 4,
            'float16': 2,
            'bfloat16': 2,
            'int8': 1,
            'int4': 0.5,
        }
        
        # Extract base dtype from quantization info
        if self.quantization and 'int8' in self.quantization:
            bytes = 1
        elif self.quantization and 'int4' in self.quantization:
            bytes = 0.5
        else:
            bytes = bytes_per_param.get(self.dtype, 2)
        
        return (self.total_params * bytes) / (1024**3)


class ModelShard:
    """Represents a shard of a model containing specific layers."""
    def __init__(self, layers: List[nn.Module], layer_indices: List[int], 
                 embed_tokens: Optional[nn.Module] = None,
                 norm: Optional[nn.Module] = None,
                 lm_head: Optional[nn.Module] = None,
                 use_tied_embeddings: bool = False):
        self.layers = layers
        self.layer_indices = layer_indices
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.lm_head = lm_head
        self.use_tied_embeddings = use_tied_embeddings
        
    def __call__(self, x: mx.array, cache: Optional[List] = None) -> mx.array:
        """Forward pass through the shard."""
        # Apply embedding if this is the first shard
        if self.embed_tokens is not None and len(self.layer_indices) > 0 and self.layer_indices[0] == 0:
            # Input should be token ids, convert to embeddings
            # Ensure input is batched: if 1D, make it 2D with batch_size=1
            if len(x.shape) == 1:
                x = mx.expand_dims(x, axis=0)  # Add batch dimension
            x = self.embed_tokens(x)
        
        # Process transformer layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x = layer(x, mask=None, cache=layer_cache)
        
        # Apply final normalization and output projection if this is the last shard
        if self.norm is not None:
            x = self.norm(x)
        
        # Apply output projection (lm_head or tied embeddings)
        if self.lm_head is not None or (self.use_tied_embeddings and self.embed_tokens is not None):
            x = self.apply_output_projection(x)
        
        return x
    
    def apply_output_projection(self, x: mx.array) -> mx.array:
        """Apply output projection (lm_head or tied embeddings)."""
        if self.lm_head is not None:
            return self.lm_head(x)
        elif self.use_tied_embeddings and self.embed_tokens is not None:
            # Use embedding weights as output projection
            return self.embed_tokens.as_linear(x)
        else:
            # No output projection, return as-is
            return x


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.config = None
        self.model_info = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the model architecture."""
        pass
    
    @abstractmethod
    def create_shards(self, num_shards: int, 
                     shard_sizes: Optional[List[float]] = None) -> List[ModelShard]:
        """Create model shards for distribution.
        
        Args:
            num_shards: Number of shards to create
            shard_sizes: Optional proportional sizes for each shard (must sum to 1.0)
            
        Returns:
            List of ModelShard objects
        """
        pass
    
    @abstractmethod
    def get_layer_size(self, layer_idx: int) -> int:
        """Get the size in bytes of a specific layer."""
        pass


class QwenModelWrapper(BaseModelWrapper):
    """Wrapper for Qwen models."""
    
    def load_model(self) -> None:
        """Load Qwen model."""
        logger.info(f"Loading Qwen model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)
        
        # Debug model structure
        logger.debug(f"Model type: {type(self.model)}")
        logger.debug(f"Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
        
        # Check for lm_head and tied embeddings
        has_lm_head = hasattr(self.model, 'lm_head')
        logger.debug(f"Model has lm_head: {has_lm_head}")
        
        # Access config - Qwen models have it at model.config
        if hasattr(self.model, 'config'):
            self.config = self.model.config
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
            self.config = self.model.model.config
        else:
            # Try to load from config.json
            self._load_config_from_file()
        
        # Debug tied embeddings setting
        if hasattr(self.model, 'args'):
            tie_embeddings = getattr(self.model.args, 'tie_word_embeddings', None)
            logger.debug(f"Model args.tie_word_embeddings: {tie_embeddings}")
        if self.config and hasattr(self.config, 'tie_word_embeddings'):
            tie_embeddings = getattr(self.config, 'tie_word_embeddings', None)
            logger.debug(f"Config tie_word_embeddings: {tie_embeddings}")
        
        self.model_info = self.get_model_info()
        logger.info(f"Loaded model info: {self.model_info}")
    
    def _load_config_from_file(self):
        """Load config from config.json file."""
        # This is a fallback for when config is not available as attribute
        pass
    
    def get_model_info(self) -> ModelInfo:
        """Get Qwen model information."""
        # Common Qwen config attributes
        num_layers = getattr(self.config, 'num_hidden_layers', 28)
        hidden_size = getattr(self.config, 'hidden_size', 2048)
        num_attention_heads = getattr(self.config, 'num_attention_heads', 16)
        num_key_value_heads = getattr(self.config, 'num_key_value_heads', num_attention_heads)
        vocab_size = getattr(self.config, 'vocab_size', 151936)
        max_position_embeddings = getattr(self.config, 'max_position_embeddings', 2048)
        
        # Estimate total parameters
        total_params = self._estimate_total_params(
            num_layers, hidden_size, num_attention_heads, 
            num_key_value_heads, vocab_size
        )
        
        # Detect quantization
        quantization = None
        if '8bit' in self.model_name or 'int8' in self.model_name:
            quantization = 'int8'
        elif '4bit' in self.model_name or 'int4' in self.model_name:
            quantization = 'int4'
        
        return ModelInfo(
            name=self.model_name,
            architecture='qwen2',
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            total_params=total_params,
            dtype='float16',  # Default for MLX
            quantization=quantization
        )
    
    def _estimate_total_params(self, num_layers: int, hidden_size: int,
                              num_attention_heads: int, num_key_value_heads: int,
                              vocab_size: int) -> int:
        """Estimate total parameters for Qwen model."""
        # Embedding parameters
        embed_params = vocab_size * hidden_size
        
        # Attention parameters per layer
        head_dim = hidden_size // num_attention_heads
        q_params = hidden_size * hidden_size
        k_params = num_key_value_heads * head_dim * hidden_size
        v_params = num_key_value_heads * head_dim * hidden_size
        o_params = hidden_size * hidden_size
        attention_params = q_params + k_params + v_params + o_params
        
        # MLP parameters per layer (Qwen uses SwiGLU)
        mlp_params = 3 * hidden_size * (4 * hidden_size)  # gate, up, down projections
        
        # Layer norm parameters
        ln_params = 2 * hidden_size  # Two layer norms per transformer block
        
        # Total for all layers
        total_layer_params = num_layers * (attention_params + mlp_params + ln_params)
        
        # Output head
        head_params = hidden_size * vocab_size
        
        return embed_params + total_layer_params + head_params
    
    def create_shards(self, num_shards: int, 
                     shard_sizes: Optional[List[float]] = None) -> List[ModelShard]:
        """Create Qwen model shards."""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Access the actual model layers
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
        
        layers = base_model.layers
        num_layers = len(layers)
        
        # Calculate layer distribution
        if shard_sizes:
            # Use provided proportions
            if len(shard_sizes) != num_shards:
                raise ValueError(f"shard_sizes length {len(shard_sizes)} != num_shards {num_shards}")
            if abs(sum(shard_sizes) - 1.0) > 0.001:
                raise ValueError(f"shard_sizes must sum to 1.0, got {sum(shard_sizes)}")
            
            # Convert proportions to layer counts
            layer_counts = []
            remaining_layers = num_layers
            for i, proportion in enumerate(shard_sizes[:-1]):
                count = int(num_layers * proportion)
                layer_counts.append(count)
                remaining_layers -= count
            layer_counts.append(remaining_layers)
        else:
            # Uniform distribution
            base_layers = num_layers // num_shards
            extra_layers = num_layers % num_shards
            layer_counts = [base_layers + (1 if i < extra_layers else 0) 
                           for i in range(num_shards)]
        
        # Create shards
        shards = []
        layer_idx = 0
        
        for shard_idx in range(num_shards):
            # Determine layers for this shard
            shard_layer_count = layer_counts[shard_idx]
            shard_layers = layers[layer_idx:layer_idx + shard_layer_count]
            shard_indices = list(range(layer_idx, layer_idx + shard_layer_count))
            
            # Determine if we're using tied embeddings
            # Check multiple possible attribute locations for tie_word_embeddings
            use_tied_embeddings = False
            if hasattr(self.model, 'args'):
                use_tied_embeddings = getattr(self.model.args, 'tie_word_embeddings', False)
            elif hasattr(self.config, 'tie_word_embeddings'):
                use_tied_embeddings = getattr(self.config, 'tie_word_embeddings', False)
            
            logger.debug(f"Model uses tied embeddings: {use_tied_embeddings}")
            
            # First shard gets embedding, or last shard if using tied embeddings
            embed_tokens = None
            if shard_idx == 0:
                embed_tokens = base_model.embed_tokens
            elif shard_idx == num_shards - 1 and use_tied_embeddings:
                # Last shard also needs embedding tokens for output projection
                embed_tokens = base_model.embed_tokens
            
            # Last shard gets norm and lm_head
            norm = base_model.norm if shard_idx == num_shards - 1 else None
            
            # Handle lm_head - check if it exists or if embeddings are tied
            lm_head = None
            if shard_idx == num_shards - 1:
                # More robust check for lm_head existence
                model_has_lm_head = False
                try:
                    lm_head_attr = getattr(self.model, 'lm_head', None)
                    if lm_head_attr is not None:
                        lm_head = lm_head_attr
                        model_has_lm_head = True
                        logger.debug("Found lm_head on model")
                except AttributeError:
                    model_has_lm_head = False
                
                if not model_has_lm_head:
                    if use_tied_embeddings:
                        # For tied embeddings, we don't have a separate lm_head
                        # The embedding layer will be used for output projection
                        logger.info("Model uses tied word embeddings - no separate lm_head")
                        lm_head = None
                    else:
                        # This might be an issue - no lm_head and no tied embeddings
                        logger.warning("No lm_head found and tie_word_embeddings not set. "
                                     "Model may use tied embeddings but it's not properly detected.")
                        lm_head = None
            
            shard = ModelShard(
                layers=shard_layers,
                layer_indices=shard_indices,
                embed_tokens=embed_tokens,
                norm=norm,
                lm_head=lm_head,
                use_tied_embeddings=use_tied_embeddings
            )
            shards.append(shard)
            
            layer_idx += shard_layer_count
            
            logger.info(f"Created shard {shard_idx}: layers {shard_indices}, "
                       f"embed={embed_tokens is not None}, "
                       f"norm={norm is not None}, "
                       f"lm_head={lm_head is not None}")
        
        return shards
    
    def get_layer_size(self, layer_idx: int) -> int:
        """Estimate size of a Qwen layer in bytes."""
        info = self.model_info
        
        # Calculate parameters in one transformer layer
        head_dim = info.hidden_size // info.num_attention_heads
        
        # Attention parameters
        q_params = info.hidden_size * info.hidden_size
        k_params = info.num_key_value_heads * head_dim * info.hidden_size
        v_params = info.num_key_value_heads * head_dim * info.hidden_size
        o_params = info.hidden_size * info.hidden_size
        attention_params = q_params + k_params + v_params + o_params
        
        # MLP parameters (SwiGLU)
        mlp_params = 3 * info.hidden_size * (4 * info.hidden_size)
        
        # Layer norm parameters
        ln_params = 2 * info.hidden_size
        
        total_params = attention_params + mlp_params + ln_params
        
        # Bytes per parameter
        if info.quantization == 'int8':
            bytes_per_param = 1
        elif info.quantization == 'int4':
            bytes_per_param = 0.5
        else:
            bytes_per_param = 2  # float16
        
        return int(total_params * bytes_per_param)


class LlamaModelWrapper(BaseModelWrapper):
    """Wrapper for Llama models."""
    
    def load_model(self) -> None:
        """Load Llama model."""
        logger.info(f"Loading Llama model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)
        
        # Similar config access as Qwen
        if hasattr(self.model, 'config'):
            self.config = self.model.config
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
            self.config = self.model.model.config
        
        self.model_info = self.get_model_info()
        logger.info(f"Loaded model info: {self.model_info}")
    
    def get_model_info(self) -> ModelInfo:
        """Get Llama model information."""
        # Llama-specific config
        num_layers = getattr(self.config, 'num_hidden_layers', 32)
        hidden_size = getattr(self.config, 'hidden_size', 4096)
        num_attention_heads = getattr(self.config, 'num_attention_heads', 32)
        num_key_value_heads = getattr(self.config, 'num_key_value_heads', num_attention_heads)
        vocab_size = getattr(self.config, 'vocab_size', 32000)
        max_position_embeddings = getattr(self.config, 'max_position_embeddings', 4096)
        
        # Similar parameter estimation
        total_params = self._estimate_total_params(
            num_layers, hidden_size, num_attention_heads, 
            num_key_value_heads, vocab_size
        )
        
        quantization = None
        if '8bit' in self.model_name or 'int8' in self.model_name:
            quantization = 'int8'
        elif '4bit' in self.model_name or 'int4' in self.model_name:
            quantization = 'int4'
        
        return ModelInfo(
            name=self.model_name,
            architecture='llama',
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            total_params=total_params,
            dtype='float16',
            quantization=quantization
        )
    
    def _estimate_total_params(self, num_layers: int, hidden_size: int,
                              num_attention_heads: int, num_key_value_heads: int,
                              vocab_size: int) -> int:
        """Estimate total parameters for Llama model."""
        # Similar to Qwen but Llama uses standard FFN (not SwiGLU)
        embed_params = vocab_size * hidden_size
        
        head_dim = hidden_size // num_attention_heads
        q_params = hidden_size * hidden_size
        k_params = num_key_value_heads * head_dim * hidden_size
        v_params = num_key_value_heads * head_dim * hidden_size
        o_params = hidden_size * hidden_size
        attention_params = q_params + k_params + v_params + o_params
        
        # Standard FFN (gate, down, up)
        mlp_params = 2 * hidden_size * (4 * hidden_size)
        
        ln_params = 2 * hidden_size
        total_layer_params = num_layers * (attention_params + mlp_params + ln_params)
        
        head_params = hidden_size * vocab_size
        
        return embed_params + total_layer_params + head_params
    
    def create_shards(self, num_shards: int, 
                     shard_sizes: Optional[List[float]] = None) -> List[ModelShard]:
        """Create Llama model shards."""
        # Implementation similar to Qwen
        return super().create_shards(num_shards, shard_sizes)
    
    def get_layer_size(self, layer_idx: int) -> int:
        """Estimate size of a Llama layer in bytes."""
        # Similar to Qwen implementation
        return 0  # Placeholder


class ModelFactory:
    """Factory for creating appropriate model wrappers."""
    
    # Map of model name patterns to wrapper classes
    MODEL_PATTERNS = {
        'qwen': QwenModelWrapper,
        'llama': LlamaModelWrapper,
        'mistral': LlamaModelWrapper,  # Mistral uses Llama architecture
        'phi': LlamaModelWrapper,      # Phi is similar to Llama
    }
    
    @classmethod
    def create_wrapper(cls, model_name: str) -> BaseModelWrapper:
        """Create appropriate wrapper for the model.
        
        Args:
            model_name: Name of the model (e.g., "mlx-community/Qwen3-1.7B-8bit")
            
        Returns:
            Appropriate model wrapper instance
        """
        model_lower = model_name.lower()
        
        for pattern, wrapper_class in cls.MODEL_PATTERNS.items():
            if pattern in model_lower:
                logger.info(f"Using {wrapper_class.__name__} for model {model_name}")
                return wrapper_class(model_name)
        
        # Default to Llama wrapper for unknown models
        logger.warning(f"Unknown model type for {model_name}, using LlamaModelWrapper")
        return LlamaModelWrapper(model_name)
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported model architectures."""
        return list(cls.MODEL_PATTERNS.keys())


if __name__ == "__main__":
    # Test model abstraction
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    
    # Create wrapper
    wrapper = ModelFactory.create_wrapper(model_name)
    
    # Load model
    wrapper.load_model()
    
    # Get model info
    info = wrapper.model_info
    print(f"\nModel Info:")
    print(f"  Architecture: {info.architecture}")
    print(f"  Layers: {info.num_layers}")
    print(f"  Hidden Size: {info.hidden_size}")
    print(f"  Total Params: {info.total_params:,}")
    print(f"  Estimated Size: {info.estimate_size_gb():.2f} GB")
    
    # Test sharding
    print(f"\nTesting uniform sharding (2 devices):")
    shards = wrapper.create_shards(2)
    for i, shard in enumerate(shards):
        print(f"  Shard {i}: {len(shard.layers)} layers, indices {shard.layer_indices}")
    
    print(f"\nTesting proportional sharding (3 devices with 50%, 30%, 20%):")
    shards = wrapper.create_shards(3, shard_sizes=[0.5, 0.3, 0.2])
    for i, shard in enumerate(shards):
        print(f"  Shard {i}: {len(shard.layers)} layers, indices {shard.layer_indices}")