"""
Sharded model loader for distributed inference
Adapted from exo's approach for MLX distributed inference
"""
import glob
import json
from pathlib import Path
from typing import Dict, Any, Optional
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import TransformerBlock, ModelArgs as QwenModelArgs
from mlx_lm.models.base import create_attention_mask
from shard import Shard


class IdentityBlock(nn.Module):
    """Pass-through block for layers not in this shard"""
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        return x


class ShardedQwen2Model(nn.Module):
    """Qwen2 model that only loads layers for its shard"""
    
    def __init__(self, args: QwenModelArgs, shard: Shard):
        super().__init__()
        self.args = args
        self.shard = shard
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        
        # Only load embeddings on first shard
        if self.shard.is_first_layer():
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        
        # Load layers - use IdentityBlock for layers not in our shard
        self.layers = []
        for i in range(self.num_hidden_layers):
            if self.shard.contains_layer(i):
                self.layers.append(TransformerBlock(args=args))
            else:
                self.layers.append(IdentityBlock())
        
        # Only load output norm and head on last shard
        if self.shard.is_last_layer():
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    
    def __call__(self, inputs: mx.array, cache=None):
        # Only apply embeddings on first shard
        if self.shard.is_first_layer():
            h = self.embed_tokens(inputs)
        else:
            h = inputs
        
        # Create attention mask if needed
        mask = None
        if h.shape[1] > 1:
            mask = create_attention_mask(h, cache)
        
        if cache is None:
            cache = [None] * len(self.layers)
        
        # Process through layers
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        
        # Only apply final norm on last shard
        if self.shard.is_last_layer():
            h = self.norm(h)
        
        return h


class ShardedQwen2(nn.Module):
    """Full sharded Qwen2 model with LM head"""
    
    def __init__(self, args: QwenModelArgs, shard: Shard):
        super().__init__()
        self.args = args
        self.shard = shard
        self.model_type = args.model_type
        self.model = ShardedQwen2Model(args, shard)
        
        # Only load LM head on last shard
        if self.shard.is_last_layer():
            if not args.tie_word_embeddings:
                self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    
    @property
    def layers(self):
        """Expose layers for cache creation"""
        return self.model.layers
    
    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        
        # Only apply LM head on last shard
        if self.shard.is_last_layer():
            if self.args.tie_word_embeddings:
                out = self.model.embed_tokens.as_linear(out)
            else:
                out = self.lm_head(out)
        
        return out
    
    def sanitize_weights(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Filter weights to only include those needed for this shard"""
        shard_weights = {}
        
        for key, value in weights.items():
            # Skip rotary embeddings
            if "self_attn.rotary_emb.inv_freq" in key:
                continue
            
            # Handle layer weights
            if key.startswith('model.layers.'):
                layer_num = int(key.split('.')[2])
                if self.shard.contains_layer(layer_num):
                    # Keep the original layer index (don't adjust)
                    shard_weights[key] = value
            
            # Handle embeddings (first shard only)
            elif self.shard.is_first_layer() and key.startswith('model.embed_tokens'):
                shard_weights[key] = value
            
            # Handle LM head (last shard only)
            elif self.shard.is_last_layer():
                if not self.args.tie_word_embeddings and key.startswith('lm_head'):
                    shard_weights[key] = value
                elif key.startswith('model.norm'):
                    shard_weights[key] = value
        
        return shard_weights


def load_sharded_model(model_path: str, shard: Shard, lazy: bool = False) -> ShardedQwen2:
    """Load a sharded Qwen2 model"""
    
    model_path = Path(model_path)
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create model args
    args = QwenModelArgs.from_dict(config)
    
    # Create sharded model
    model = ShardedQwen2(args, shard)
    
    # Find weight files
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    
    # Load all weights
    weights = {}
    for wf in sorted(weight_files):
        weights.update(mx.load(wf))
    
    # Filter weights for this shard
    shard_weights = model.sanitize_weights(weights)
    
    # Load filtered weights into model
    model.load_weights(list(shard_weights.items()), strict=False)
    
    if not lazy:
        mx.eval(model.parameters())
    
    model.eval()
    return model