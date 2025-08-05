#!/usr/bin/env python3
"""
Compare single-node vs distributed inference logits to verify numerical consistency.

This script:
1. Runs the same prompt through both single-node and distributed inference
2. Compares logits at each generation step
3. Reports maximum absolute differences
4. Fails if max(|Δ|) > 1e-3 for any position
"""

import asyncio
import mlx.core as mx
from mlx_lm import load
import numpy as np
import argparse
import json
import sys
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_single_node_inference(model_name: str, prompt: str, max_tokens: int = 20) -> Tuple[List[mx.array], List[int]]:
    """Run inference on a single device and collect logits"""
    logger.info("Loading model for single-node inference...")
    model, tokenizer = load(model_name)
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    logger.info(f"Prompt tokens: {tokens}")
    
    # Generate tokens and collect logits
    all_logits = []
    generated_tokens = []
    
    # Initialize cache
    cache = None
    
    for i in range(max_tokens):
        # Prepare input
        if i == 0:
            # First pass: process full prompt
            input_ids = mx.array([tokens])
        else:
            # Subsequent passes: process only new token
            input_ids = mx.array([[generated_tokens[-1]]])
        
        # Forward pass
        logits, cache = model(input_ids, cache=cache)
        
        # Get logits for last position
        logits_last = logits[0, -1, :]
        
        # Store logits (convert to float32 for comparison)
        logits_f32 = logits_last.astype(mx.float32)
        mx.eval(logits_f32)
        all_logits.append(logits_f32)
        
        # Sample next token (using same temperature as distributed)
        temperature = 0.7
        logits_scaled = logits_last / temperature
        probs = mx.softmax(logits_scaled)
        token = mx.argmax(probs).item()
        
        generated_tokens.append(token)
        logger.info(f"Single-node token {i}: {token} ({tokenizer.decode([token]) if hasattr(tokenizer, 'decode') else 'N/A'})")
    
    return all_logits, generated_tokens

async def run_distributed_inference(prompt: str, max_tokens: int = 20, port: int = 50051) -> Tuple[List[np.ndarray], List[int]]:
    """Run distributed inference and collect logits"""
    import aiohttp
    
    api_url = f"http://localhost:{port}/v1/chat/completions"
    
    # Prepare request
    request_data = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
        "return_logits": True  # Request logits to be returned
    }
    
    logger.info("Running distributed inference...")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json=request_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed: {response.status} - {error_text}")
            
            result = await response.json()
    
    # Extract logits and tokens from response
    if "logits" not in result:
        # If API doesn't return logits, we need to modify the API server
        # For now, we'll need to run a modified version
        raise Exception("API server must be configured to return logits for comparison")
    
    logits = result["logits"]
    tokens = result["tokens"]
    
    return logits, tokens

def compare_logits(single_logits: List[mx.array], distributed_logits: List[np.ndarray], tolerance: float = 1e-3) -> bool:
    """Compare logits from single-node and distributed inference"""
    if len(single_logits) != len(distributed_logits):
        logger.error(f"Length mismatch: single={len(single_logits)}, distributed={len(distributed_logits)}")
        return False
    
    all_passed = True
    max_diffs = []
    
    for pos in range(len(single_logits)):
        # Convert single-node logits to numpy
        single_np = np.array(single_logits[pos], dtype=np.float32)
        distributed_np = np.array(distributed_logits[pos], dtype=np.float32)
        
        # Check shapes
        if single_np.shape != distributed_np.shape:
            logger.error(f"Position {pos}: Shape mismatch - single={single_np.shape}, distributed={distributed_np.shape}")
            all_passed = False
            continue
        
        # Compute differences
        diff = np.abs(single_np - distributed_np)
        max_diff = np.max(diff)
        max_diffs.append(max_diff)
        
        # Find indices of largest differences
        largest_diff_indices = np.argsort(diff)[-5:][::-1]
        
        # Report results
        status = "PASS" if max_diff < tolerance else "FAIL"
        logger.info(f"Position {pos}: max|Δ| = {max_diff:.6f} [{status}]")
        
        if max_diff >= tolerance:
            all_passed = False
            logger.error(f"  Tolerance exceeded! Expected < {tolerance}")
            logger.error(f"  Top 5 differences:")
            for idx in largest_diff_indices:
                logger.error(f"    Token {idx}: single={single_np[idx]:.6f}, distributed={distributed_np[idx]:.6f}, |Δ|={diff[idx]:.6f}")
        else:
            logger.debug(f"  Within tolerance (< {tolerance})")
    
    # Summary statistics
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Positions compared: {len(single_logits)}")
    logger.info(f"Max difference across all positions: {max(max_diffs):.6f}")
    logger.info(f"Mean of max differences: {np.mean(max_diffs):.6f}")
    logger.info(f"Positions exceeding tolerance: {sum(1 for d in max_diffs if d >= tolerance)}")
    
    return all_passed

async def main():
    parser = argparse.ArgumentParser(description="Compare single-node vs distributed inference")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--prompt", default="Once upon a time, there was a", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Maximum acceptable difference")
    parser.add_argument("--port", type=int, default=50051, help="API server port for distributed inference")
    parser.add_argument("--single-only", action="store_true", help="Run only single-node inference")
    parser.add_argument("--distributed-only", action="store_true", help="Run only distributed inference")
    
    args = parser.parse_args()
    
    try:
        if args.single_only:
            # Run only single-node
            single_logits, single_tokens = await run_single_node_inference(args.model, args.prompt, args.max_tokens)
            logger.info(f"Single-node generated {len(single_tokens)} tokens")
            return
        
        if args.distributed_only:
            # Run only distributed
            distributed_logits, distributed_tokens = await run_distributed_inference(args.prompt, args.max_tokens, args.port)
            logger.info(f"Distributed generated {len(distributed_tokens)} tokens")
            return
        
        # Run both and compare
        logger.info("=== Running Single-Node Inference ===")
        single_logits, single_tokens = await run_single_node_inference(args.model, args.prompt, args.max_tokens)
        
        logger.info("\n=== Running Distributed Inference ===")
        distributed_logits, distributed_tokens = await run_distributed_inference(args.prompt, args.max_tokens, args.port)
        
        logger.info("\n=== Comparing Logits ===")
        passed = compare_logits(single_logits, distributed_logits, args.tolerance)
        
        if passed:
            logger.info("\n✅ ALL TESTS PASSED! Logits are within tolerance.")
            sys.exit(0)
        else:
            logger.error("\n❌ TESTS FAILED! Logits exceed tolerance.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())