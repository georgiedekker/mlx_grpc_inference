#!/usr/bin/env python3
"""
Distributed Model Inference using Hybrid Communication
"""
import torch
import torch.nn as nn
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
import time

from src.hybrid import HybridDistributedInference

logger = logging.getLogger(__name__)

class DistributedModelInference:
    """
    Handles distributed model inference across multiple nodes
    using hybrid communication strategies
    """
    
    def __init__(self,
                 model: nn.Module,
                 hybrid_dist: HybridDistributedInference,
                 kv_cache: Optional[Any] = None,
                 assigned_layers: Optional[List[int]] = None):
        
        self.model = model
        self.hybrid_dist = hybrid_dist
        self.kv_cache = kv_cache
        self.assigned_layers = assigned_layers or []
        self.rank = hybrid_dist.rank
        self.world_size = hybrid_dist.world_size
        
        # Performance tracking
        self.transfer_times = []
        self.compute_times = []
        
        logger.info(f"Initialized distributed inference on rank {self.rank}")
    
    async def generate(self,
                      input_ids: torch.Tensor,
                      max_length: int = 100,
                      temperature: float = 1.0,
                      do_sample: bool = True,
                      top_k: int = 50,
                      top_p: float = 0.95) -> torch.Tensor:
        """
        Generate text using distributed model
        """
        generated_ids = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_length - input_ids.shape[1]):
            # Run forward pass
            logits, past_key_values = await self.distributed_forward(
                generated_ids[:, -1:] if past_key_values else generated_ids,
                past_key_values=past_key_values
            )
            
            # Sample next token (only on coordinator)
            if self.rank == 0:
                next_token = self._sample_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # Broadcast next token to all ranks
                next_token = self.hybrid_dist.broadcast_tensor(next_token, src=0)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            else:
                # Workers receive the next token
                next_token = torch.zeros(1, 1, dtype=torch.long)
                next_token = self.hybrid_dist.broadcast_tensor(next_token, src=0)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS token
            if next_token.item() == self.model.config.eos_token_id:
                break
        
        return generated_ids
    
    async def distributed_forward(self,
                                 input_ids: torch.Tensor,
                                 past_key_values: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Distributed forward pass through the model
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Embeddings (rank 0 only)
        if self.rank == 0:
            hidden_states = self.model.embed_tokens(input_ids)
            # Send to next rank if multi-node
            if self.world_size > 1:
                await self._transfer_hidden_states(hidden_states, dest_rank=1)
        else:
            # Receive from previous rank
            hidden_states = await self._receive_hidden_states(from_rank=self.rank - 1)
        
        # Process assigned layers
        layer_outputs = past_key_values if past_key_values else []
        
        for layer_idx in self.assigned_layers:
            start_time = time.time()
            
            # Get layer
            layer = self.model.layers[layer_idx]
            
            # Apply layer
            if past_key_values and len(past_key_values) > layer_idx:
                layer_past = past_key_values[layer_idx]
            else:
                layer_past = None
            
            layer_output = layer(
                hidden_states,
                past_key_value=layer_past,
                use_cache=True
            )
            
            hidden_states = layer_output[0]
            
            # Store KV cache
            if len(layer_output) > 1:
                layer_outputs.append(layer_output[1])
            
            self.compute_times.append(time.time() - start_time)
        
        # Transfer to next rank if needed
        if self.rank < self.world_size - 1 and self.assigned_layers:
            await self._transfer_hidden_states(hidden_states, dest_rank=self.rank + 1)
        
        # Final processing on last rank
        if self.rank == self.world_size - 1:
            # Apply final norm
            hidden_states = self.model.norm(hidden_states)
            
            # Get logits
            logits = self.model.lm_head(hidden_states)
            
            # Send logits back to coordinator
            if self.rank != 0:
                await self._transfer_logits(logits, dest_rank=0)
            
            return logits, tuple(layer_outputs)
        
        # Coordinator receives final logits
        if self.rank == 0 and self.world_size > 1:
            logits = await self._receive_logits(from_rank=self.world_size - 1)
            return logits, tuple(layer_outputs)
        
        # Intermediate ranks
        return None, tuple(layer_outputs)
    
    async def _transfer_hidden_states(self, hidden_states: torch.Tensor, dest_rank: int):
        """Transfer hidden states to next rank"""
        start_time = time.time()
        await self.hybrid_dist.transfer_tensor(hidden_states, dest_rank)
        self.transfer_times.append(time.time() - start_time)
        logger.debug(f"Transferred hidden states to rank {dest_rank}")
    
    async def _receive_hidden_states(self, from_rank: int) -> torch.Tensor:
        """Receive hidden states from previous rank"""
        # This is a simplified version - in practice we'd need proper receiving logic
        # based on the transfer method used
        logger.debug(f"Waiting for hidden states from rank {from_rank}")
        # Placeholder - actual implementation would use the hybrid system
        return torch.zeros(1, 1, 768)  # Dummy tensor
    
    async def _transfer_logits(self, logits: torch.Tensor, dest_rank: int):
        """Transfer logits to coordinator"""
        await self.hybrid_dist.transfer_tensor(logits, dest_rank)
        logger.debug(f"Transferred logits to rank {dest_rank}")
    
    async def _receive_logits(self, from_rank: int) -> torch.Tensor:
        """Receive logits from last rank"""
        logger.debug(f"Waiting for logits from rank {from_rank}")
        # Placeholder - actual implementation would use the hybrid system
        return torch.zeros(1, 1, 50000)  # Dummy tensor
    
    def _sample_token(self,
                     logits: torch.Tensor,
                     temperature: float = 1.0,
                     do_sample: bool = True,
                     top_k: int = 50,
                     top_p: float = 0.95) -> torch.Tensor:
        """Sample next token from logits"""
        if temperature != 1.0:
            logits = logits / temperature
        
        if not do_sample:
            # Greedy decoding
            return logits.argmax(dim=-1)
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        stats = {}
        
        if self.compute_times:
            stats['avg_compute_time_ms'] = sum(self.compute_times) / len(self.compute_times) * 1000
        
        if self.transfer_times:
            stats['avg_transfer_time_ms'] = sum(self.transfer_times) / len(self.transfer_times) * 1000
        
        stats['total_layers'] = len(self.assigned_layers)
        
        return stats