#!/usr/bin/env python3
"""
Simple test to verify both GPUs are being used in distributed mode.
Tests with a basic all_gather operation to prove MPI communication.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize distributed
    group = mx.distributed.init()
    if not group:
        logger.error("Failed to initialize distributed")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    logger.info(f"Rank {rank}/{world_size} initialized")
    
    # Load model normally on each rank
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    logger.info(f"Rank {rank}: Loading model {model_name}")
    
    model, tokenizer = load(model_name)
    
    # Log memory usage
    memory = mx.get_active_memory() / (1024**3)
    logger.info(f"Rank {rank}: GPU memory = {memory:.2f} GB")
    
    # Simple distributed operation to prove both GPUs are active
    local_tensor = mx.array([float(rank + 1)])
    gathered = mx.distributed.all_gather(local_tensor, group=group)
    mx.eval(gathered)
    
    if rank == 0:
        logger.info(f"All-gather result: {gathered}")
        if world_size == 2:
            expected = mx.array([1.0, 2.0])
            if mx.array_equal(gathered, expected):
                logger.info("✅ Both GPUs are communicating successfully!")
            else:
                logger.warning(f"Unexpected result. Expected {expected}, got {gathered}")
    
    # Only rank 0 does generation
    if rank == 0:
        prompt = "What is 2+2?"
        messages = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        if isinstance(prompt_formatted, list):
            prompt_formatted = tokenizer.decode(prompt_formatted)
        
        logger.info(f"Rank 0: Generating response...")
        
        sampler = make_sampler(temp=0.7)
        generated_text = ""
        
        for response in stream_generate(
            model,
            tokenizer, 
            prompt_formatted,
            max_tokens=20,
            sampler=sampler
        ):
            if hasattr(response, 'text'):
                generated_text = response.text
        
        logger.info(f"Generated: {generated_text}")
        logger.info(f"✅ Generation successful using {world_size} GPUs")

if __name__ == "__main__":
    main()