#\!/usr/bin/env python3
"""
Native pipeline support for Qwen3 following DeepSeek-V3 pattern
Based on MLX-LM's deepseek_v3.py pipeline implementation
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

def add_native_pipeline_to_qwen3(model):
    """
    Add native pipeline support to Qwen3 model following DeepSeek-V3 pattern
    This properly divides layers and handles communication
    """
    
    # Store original forward for fallback
    model.model._original_forward = model.model.__call__
    
    def pipeline(self, group):
        """
        Pipeline parallelism implementation following DeepSeek-V3 pattern
        Divides layers across ranks and sets up communication
        """
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        
        # Calculate layer distribution
        n_layers = len(self.layers)
        layers_per_rank = n_layers // self.pipeline_size
        remainder = n_layers % self.pipeline_size
        
        # Distribute remainder layers to first ranks
        if self.pipeline_rank < remainder:
            self.start_idx = self.pipeline_rank * (layers_per_rank + 1)
            self.end_idx = self.start_idx + layers_per_rank + 1
        else:
            self.start_idx = self.pipeline_rank * layers_per_rank + remainder
            self.end_idx = self.start_idx + layers_per_rank
        
        # Important: Keep references to all layers but only process our slice
        self.pipeline_layers = self.layers[self.start_idx:self.end_idx]
        
        logger.info(f"Pipeline rank {self.pipeline_rank}: Processing layers {self.start_idx}-{self.end_idx-1} (total: {n_layers})")
        
        # Store group for communication
        self.pipeline_group = group
    
    def pipelined_forward(self, inputs: mx.array, mask=None, cache=None, input_embeddings=None):
        """
        Pipelined forward pass with proper send/recv communication
        Following DeepSeek-V3's approach but adapted for Qwen3
        """
        
        # Check if we're in pipeline mode
        if not hasattr(self, 'pipeline_size') or self.pipeline_size == 1:
            # Fallback to original forward
            return self._original_forward(inputs, mask, cache, input_embeddings)
        
        # Handle input embeddings
        if self.pipeline_rank == self.pipeline_size - 1:  # First stage (processes first layers)
            # Only first stage does embedding
            if input_embeddings is not None:
                h = input_embeddings
            else:
                h = self.embed_tokens(inputs)
            
            # Create mask if needed (only first stage)
            if mask is None and h.shape[1] > 1:
                from mlx_lm.models.qwen3 import create_attention_mask
                mask = create_attention_mask(h, cache)
        else:
            # Other stages receive hidden states
            # Receive from previous stage (higher rank number)
            h = mx.distributed.recv(src=self.pipeline_rank + 1, group=self.pipeline_group)
            mx.eval(h)
            
            # Also receive mask if applicable
            if mask is None:
                mask_shape = mx.distributed.recv(src=self.pipeline_rank + 1, group=self.pipeline_group)
                mx.eval(mask_shape)
                if mask_shape.item() > 0:
                    mask = mx.distributed.recv(src=self.pipeline_rank + 1, group=self.pipeline_group)
                    mx.eval(mask)
        
        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)
        
        # Process our layers
        for i, layer in enumerate(self.pipeline_layers):
            layer_idx = self.start_idx + i
            h = layer(h, mask, cache[layer_idx] if cache else None)
        
        # Send to next stage or return final output
        if self.pipeline_rank > 0:  # Not the last stage
            # Send hidden states to next stage (lower rank number)
            mx.distributed.send(h, dst=self.pipeline_rank - 1, group=self.pipeline_group)
            mx.eval(h)
            
            # Send mask info
            if mask is not None:
                mask_indicator = mx.array([1])
                mx.distributed.send(mask_indicator, dst=self.pipeline_rank - 1, group=self.pipeline_group)
                mx.eval(mask_indicator)
                mx.distributed.send(mask, dst=self.pipeline_rank - 1, group=self.pipeline_group)
                mx.eval(mask)
            else:
                mask_indicator = mx.array([0])
                mx.distributed.send(mask_indicator, dst=self.pipeline_rank - 1, group=self.pipeline_group)
                mx.eval(mask_indicator)
            
            # Return dummy output for non-final stages
            return mx.zeros_like(h)
        else:  # Last stage (rank 0)
            # Apply final normalization
            return self.norm(h)
    
    # Bind methods to model
    import types
    model.model.pipeline = types.MethodType(pipeline, model.model)
    model.model.__call__ = types.MethodType(pipelined_forward, model.model)
    
    logger.info(f"✅ Added native pipeline support to {model.model.__class__.__name__}")
    
    return model


def coordinated_generate_with_pipeline(model, tokenizer, prompt, max_tokens=100, temperature=0.7, rank=0, world_size=1):
    """
    Generate text using native pipeline parallelism
    All ranks participate simultaneously in each forward pass
    """
    from mlx_lm.sample_utils import make_sampler
    import time
    
    logger.info(f"Rank {rank}: Starting coordinated generation")
    
    # All ranks tokenize (for simplicity)
    input_ids = tokenizer.encode(prompt)
    inputs = mx.array([input_ids])
    
    # Create sampler (all ranks need same sampler for consistency)
    sampler = make_sampler(temp=temperature)
    
    generated_tokens = []
    cache = None
    
    start_time = time.time()
    
    for i in range(max_tokens):
        # All ranks do forward pass simultaneously
        # The pipelined_forward handles communication
        logits = model.model(inputs, cache=cache)
        mx.eval(logits)
        
        # Only rank 0 (final stage) gets real logits
        if rank == 0:
            # Sample next token
            next_logits = logits[:, -1, :]
            next_token = sampler(next_logits)
            token_id = int(next_token.item())
            
            # Check for EOS
            if token_id == tokenizer.eos_token_id:
                # Broadcast stop signal
                stop_signal = mx.array([1])
                for r in range(1, world_size):
                    mx.distributed.send(stop_signal, dst=r, group=model.model.pipeline_group)
                    mx.eval(stop_signal)
                break
            
            generated_tokens.append(token_id)
            
            # Broadcast next token to all ranks
            token_broadcast = mx.array([token_id])
            for r in range(1, world_size):
                mx.distributed.send(token_broadcast, dst=r, group=model.model.pipeline_group)
                mx.eval(token_broadcast)
            
            # Prepare next input
            inputs = mx.array([[token_id]])
        else:
            # Other ranks receive token or stop signal
            signal = mx.distributed.recv(src=0, group=model.model.pipeline_group)
            mx.eval(signal)
            
            if signal.item() == 1:  # Stop signal
                break
            
            token_id = int(signal.item())
            # Prepare same input as rank 0
            inputs = mx.array([[token_id]])
    
    # Return results
    if rank == 0:
        generated_text = tokenizer.decode(generated_tokens)
        gen_time = time.time() - start_time
        tokens_per_second = len(generated_tokens) / gen_time if gen_time > 0 else 0
        
        logger.info(f"✅ Generated {len(generated_tokens)} tokens in {gen_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
        
        return {
            "text": generated_text,
            "tokens": generated_tokens,
            "tokens_per_second": tokens_per_second,
            "time": gen_time
        }
    else:
        return {}


if __name__ == "__main__":
    # Test the pipeline implementation
    from mlx_lm import load
    import mlx.core as mx
    
    # Initialize distributed
    group = mx.distributed.init()
    
    if group:
        rank = group.rank()
        world_size = group.size()
        
        print(f"Rank {rank}/{world_size}: Loading model...")
        
        # Load model
        model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
        
        # Add native pipeline support
        model = add_native_pipeline_to_qwen3(model)
        
        # Apply pipeline
        model.model.pipeline(group)
        
        # Test generation
        if rank == 0:
            prompt = "What is 2+2?"
        else:
            prompt = ""  # Other ranks don't need prompt
        
        result = coordinated_generate_with_pipeline(
            model, tokenizer, prompt, 
            max_tokens=10, temperature=0.7,
            rank=rank, world_size=world_size
        )
        
        if rank == 0:
            print(f"Generated: {result['text']}")
            print(f"Speed: {result['tokens_per_second']:.1f} tokens/sec")
    else:
        print("No distributed group found. Run with mpirun -n 2")