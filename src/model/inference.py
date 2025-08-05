"""
Local inference execution for model layers.
"""

import logging
import time
from typing import List, Dict, Optional, Any, Tuple
import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class LayerProcessor:
    """Processes specific layers of a model."""
    
    def __init__(self, model: nn.Module, device_id: str, assigned_layers: List[int]):
        """
        Initialize layer processor.
        
        Args:
            model: The model containing layers
            device_id: ID of this device
            assigned_layers: List of layer indices this device should process
        """
        self.model = model
        self.device_id = device_id
        self.assigned_layers = set(assigned_layers)
        self.cache = {}
        
        logger.info(f"LayerProcessor for {device_id} handling layers: {sorted(assigned_layers)}")
    
    def process(self, 
                input_tensor: mx.array, 
                layer_indices: List[int],
                context: Optional[Dict[str, Any]] = None) -> mx.array:
        """
        Process specified layers.
        
        Args:
            input_tensor: Input hidden states
            layer_indices: Indices of layers to process
            context: Optional context (attention masks, etc.)
            
        Returns:
            Output tensor after processing layers
        """
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
                
                # Evaluate to ensure computation happens
                mx.eval(hidden_states)
                
                process_time = (time.time() - start_time) * 1000
                logger.debug(f"Layer {layer_idx} processed in {process_time:.1f}ms")
            else:
                logger.warning(f"Layer {layer_idx} not found in model")
        
        return hidden_states
    
    def _process_single_layer(self, 
                             layer: nn.Module, 
                             hidden_states: mx.array,
                             layer_idx: int,
                             context: Optional[Dict[str, Any]]) -> mx.array:
        """Process a single transformer layer."""
        # Call the layer directly - it knows how to process itself
        # Most modern models (including Qwen3) handle their own internal processing
        layer_output = layer(hidden_states)
        
        # Handle if layer returns tuple (hidden_states, attention_weights, ...)
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
        else:
            hidden_states = layer_output
        
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
        else:
            raise ValueError("Model does not have lm_head")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        # This would use MLX-specific memory tracking
        return {
            'allocated_gb': 0.0,  # Placeholder
            'cached_gb': 0.0,     # Placeholder
            'reserved_gb': 0.0    # Placeholder
        }