"""
Fixed layer processor that maintains dtype correctly.
"""

import time
import logging
from typing import List, Dict, Optional, Any
import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class FixedLayerProcessor:
    """
    Fixed layer processor that maintains dtype during processing.
    """
    
    def __init__(self, model: nn.Module, device_id: str, assigned_layers: List[int]):
        """
        Initialize the layer processor.
        
        Args:
            model: The MLX model
            device_id: Device identifier
            assigned_layers: List of layer indices assigned to this device
        """
        self.model = model
        self.device_id = device_id
        self.assigned_layers = set(assigned_layers)
        logger.info(f"LayerProcessor for {device_id} handling layers: {assigned_layers}")
    
    def process(self, 
                input_tensor: mx.array, 
                layer_indices: List[int],
                context: Optional[Dict[str, Any]] = None) -> mx.array:
        """
        Process specified layers with dtype preservation.
        
        Args:
            input_tensor: Input hidden states
            layer_indices: Indices of layers to process
            context: Optional context dict with attention masks, etc.
            
        Returns:
            Output hidden states with original dtype preserved
        """
        # Store original dtype
        original_dtype = input_tensor.dtype
        logger.info(f"Processing layers {layer_indices} with input dtype: {original_dtype}")
        
        # Validate layers
        for idx in layer_indices:
            if idx not in self.assigned_layers:
                raise ValueError(f"Layer {idx} not assigned to device {self.device_id}")
        
        # Process layers sequentially
        hidden_states = input_tensor
        
        for layer_idx in sorted(layer_indices):
            start_time = time.time()
            
            if hasattr(self.model.model, 'layers') and layer_idx < len(self.model.model.layers):
                layer = self.model.model.layers[layer_idx]
                
                # Process through layer
                hidden_states = self._process_single_layer(
                    layer, hidden_states, layer_idx, context
                )
                
                # Ensure dtype is preserved after each layer
                if hidden_states.dtype != original_dtype:
                    logger.warning(f"Layer {layer_idx} changed dtype from {original_dtype} to {hidden_states.dtype}, restoring...")
                    hidden_states = hidden_states.astype(original_dtype)
                
                # Evaluate to ensure computation happens
                mx.eval(hidden_states)
                
                process_time = (time.time() - start_time) * 1000
                logger.debug(f"Layer {layer_idx} processed in {process_time:.1f}ms")
            else:
                logger.warning(f"Layer {layer_idx} not found in model")
        
        # Final dtype check
        if hidden_states.dtype != original_dtype:
            logger.info(f"Restoring output dtype from {hidden_states.dtype} to {original_dtype}")
            hidden_states = hidden_states.astype(original_dtype)
        
        logger.info(f"Output dtype: {hidden_states.dtype}, shape: {hidden_states.shape}")
        return hidden_states
    
    def _process_single_layer(self, 
                             layer: nn.Module, 
                             hidden_states: mx.array,
                             layer_idx: int,
                             context: Optional[Dict[str, Any]]) -> mx.array:
        """Process a single transformer layer with proper layer norm handling."""
        # Store original dtype
        original_dtype = hidden_states.dtype
        
        residual = hidden_states
        
        # Input layer norm
        if hasattr(layer, 'input_layernorm'):
            hidden_states = layer.input_layernorm(hidden_states)
        
        # Self-attention
        if hasattr(layer, 'self_attn'):
            attn_output = layer.self_attn(hidden_states)
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            hidden_states = residual + attn_output
            residual = hidden_states
        
        # Post-attention layer norm
        if hasattr(layer, 'post_attention_layernorm'):
            hidden_states = layer.post_attention_layernorm(hidden_states)
        
        # MLP
        if hasattr(layer, 'mlp'):
            mlp_output = layer.mlp(hidden_states)
            hidden_states = residual + mlp_output
        
        # Ensure dtype is preserved
        if hidden_states.dtype != original_dtype:
            hidden_states = hidden_states.astype(original_dtype)
        
        return hidden_states
    
    def process_embedding(self, input_ids: mx.array) -> mx.array:
        """Process input embeddings (for first device)."""
        if hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens(input_ids)
        else:
            raise ValueError("Model does not have embed_tokens layer")
    
    def process_output(self, hidden_states: mx.array) -> mx.array:
        """Process final output layers (for last device)."""
        # Final layer norm
        if hasattr(self.model.model, 'norm'):
            hidden_states = self.model.model.norm(hidden_states)
        
        # LM head
        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(hidden_states)
            return logits
        elif hasattr(self.model.model, 'embed_tokens'):
            # For models with tied embeddings
            logits = self.model.model.embed_tokens.as_linear(hidden_states)
            return logits
        else:
            raise ValueError("Model does not have lm_head or tied embeddings")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        return {
            'allocated_gb': 0.0,
            'cached_gb': 0.0,
            'reserved_gb': 0.0
        }