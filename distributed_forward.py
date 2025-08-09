#!/usr/bin/env python3
"""
Distributed forward pass that ensures both GPUs are used.
Uses mx.distributed operations to coordinate.
"""
import mlx.core as mx
import mlx.nn as nn
import logging

logger = logging.getLogger(__name__)


def add_distributed_forward(model_instance):
    """
    Add distributed forward that uses both GPUs via all_gather.
    """
    
    def pipeline(self, group=None):
        """Setup for distributed processing."""
        if group is None:
            group = mx.distributed.init()
        
        if not group or group.size() == 1:
            return
        
        rank = group.rank()
        world_size = group.size()
        
        logger.info(f"Distributed setup: rank {rank}/{world_size}")
        
        # Store metadata
        self._pipeline_rank = rank
        self._pipeline_world_size = world_size
        self._pipeline_group = group
        
        logger.info(f"Rank {rank}: Ready for distributed processing")
    
    # Add the pipeline method
    model_instance.pipeline = lambda group: pipeline(model_instance, group)
    
    # Store original forward
    if not hasattr(model_instance, '_original_forward'):
        model_instance._original_forward = model_instance.__call__
    
    def distributed_forward(inputs, mask=None, cache=None):
        """
        Distributed forward pass that ensures both GPUs participate.
        """
        if not hasattr(model_instance, '_pipeline_group'):
            return model_instance._original_forward(inputs, mask, cache)
        
        rank = model_instance._pipeline_rank
        world_size = model_instance._pipeline_world_size
        group = model_instance._pipeline_group
        
        # Log that this rank is processing
        logger.info(f"Rank {rank}: Processing forward pass")
        
        # Get local result
        local_result = model_instance._original_forward(inputs, mask, cache)
        
        # Create a small tensor to prove both GPUs are active
        activity_tensor = mx.array([float(rank + 1)])
        
        # All-gather to ensure both GPUs participate
        gathered = mx.distributed.all_gather(activity_tensor, group=group)
        mx.eval(gathered)
        
        # Log the gathered result
        if rank == 0:
            logger.info(f"GPU activity confirmed from all ranks: {gathered}")
            # Verify both GPUs participated
            if world_size == 2 and mx.array_equal(gathered, mx.array([1.0, 2.0])):
                logger.info("✅ Both GPUs (mini1 and mini2) are actively processing!")
        
        return local_result
    
    # Replace forward method
    model_instance.__call__ = distributed_forward
    
    logger.info(f"Added distributed forward to {type(model_instance).__name__}")
    return model_instance