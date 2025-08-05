#!/usr/bin/env python3
"""
Simple distributed inference using MLX's native distributed capabilities.
This is the basic version to get working first.
"""

import mlx.core as mx
import mlx.distributed as dist
from mlx_lm import load, generate
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-8bit")
    parser.add_argument("--prompt", default="Hello! How are you?")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()
    
    # Initialize distributed - MLX will handle the communication
    world = dist.init()
    logger.info(f"Initialized distributed - rank {world.rank()}/{world.size()}")
    
    # Load model on rank 0 first
    if world.rank() == 0:
        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load(args.model)
        logger.info(f"Model loaded with {len(model.model.layers)} layers")
        
        # Test single device inference first to verify model works
        logger.info("Testing single device inference...")
        test_response = generate(
            model, tokenizer, 
            prompt="Hi", 
            max_tokens=20,
            temp=0.7
        )
        logger.info(f"Single device test: {test_response}")
        
        # Now let's try distributed inference
        # For now, just run the full model on rank 0 to establish baseline
        logger.info(f"\nProcessing prompt: {args.prompt}")
        response = generate(
            model, tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temp=args.temp
        )
        logger.info(f"Response: {response}")
    else:
        # Workers wait
        logger.info(f"Worker {world.rank()} ready")
    
    # Cleanup
    dist.cleanup()

if __name__ == "__main__":
    main()