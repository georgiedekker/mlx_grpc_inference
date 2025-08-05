#!/usr/bin/env python3
"""
Direct comparison of single-node vs distributed inference logits.

This script directly instantiates the servers to access raw logits without API modifications.
"""

import asyncio
import mlx.core as mx
import numpy as np
import argparse
import sys
import logging
from typing import List, Tuple, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_single_node_generation(model_name: str, prompt_tokens: List[int], max_tokens: int = 20) -> Tuple[List[mx.array], List[int]]:
    """Run single-node inference and collect logits at each step"""
    from mlx_lm import load
    
    logger.info("Loading model for single-node inference...")
    model, tokenizer = load(model_name)
    
    # Generate tokens and collect logits
    all_logits = []
    generated = prompt_tokens.copy()
    cache = None
    
    logger.info(f"Starting generation with prompt tokens: {prompt_tokens}")
    
    for i in range(max_tokens):
        # Prepare input
        if i == 0:
            # First pass: process full prompt
            input_ids = mx.array([generated])
        else:
            # Subsequent passes: process only new token
            input_ids = mx.array([[generated[-1]]])
        
        # Forward pass
        output = model(input_ids, cache=cache)
        
        # Handle different model return types
        if isinstance(output, tuple):
            logits, cache = output
        else:
            # Some models only return logits
            logits = output
            # Cache is managed internally
        
        # Get logits for last position
        logits_last = logits[0, -1, :]
        
        # Store raw logits
        mx.eval(logits_last)
        all_logits.append(logits_last)
        
        # Sample next token (using same method as distributed)
        temperature = 0.7
        logits_scaled = logits_last / temperature
        probs = mx.softmax(logits_scaled)
        token = mx.argmax(probs).item()
        
        generated.append(token)
        decoded = tokenizer.decode([token]) if hasattr(tokenizer, 'decode') else str(token)
        logger.info(f"Single-node token {i}: {token} ('{decoded}')")
    
    return all_logits, generated[len(prompt_tokens):]

async def run_distributed_generation(model_name: str, prompt_tokens: List[int], max_tokens: int = 20) -> Tuple[List[mx.array], List[int]]:
    """Run distributed inference using the DistributedServer directly"""
    import os
    from server_distributed import DistributedServer
    
    # Set environment variables for distributed server
    os.environ['MODEL_NAME'] = model_name
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = 'localhost'  # Force localhost for testing
    os.environ['LOCAL_TEST'] = 'false'  # Use localhost for worker connections
    
    logger.info("Starting distributed server...")
    server = DistributedServer()
    
    # Initialize the server
    await server.initialize()
    
    # Generate tokens and collect logits
    all_logits = []
    generated = prompt_tokens.copy()
    
    logger.info(f"Starting distributed generation with prompt tokens: {prompt_tokens}")
    
    for i in range(max_tokens):
        # Prepare input
        if i == 0:
            # First pass: process full prompt
            input_array = mx.array([generated], dtype=mx.int32)
        else:
            # Subsequent passes: process only new token
            input_array = mx.array([[generated[-1]]], dtype=mx.int32)
        
        # Run distributed forward pass
        logits, _ = await server.distributed_forward(input_array, mask=None, cache=server.persistent_kv_cache)
        
        # Get logits for last position
        logits_last = logits[0, -1, :]
        
        # Store raw logits
        mx.eval(logits_last)
        all_logits.append(logits_last)
        
        # Sample next token (same as single-node)
        temperature = 0.7
        logits_scaled = logits_last / temperature
        probs = mx.softmax(logits_scaled)
        token = mx.argmax(probs).item()
        
        generated.append(token)
        logger.info(f"Distributed token {i}: {token}")
    
    # Cleanup
    await server.cleanup()
    
    return all_logits, generated[len(prompt_tokens):]

def compare_logits_detailed(single_logits: List[mx.array], distributed_logits: List[mx.array], tolerance: float = 1e-3) -> Dict[str, Any]:
    """Compare logits with detailed statistics"""
    results = {
        "passed": True,
        "positions": len(single_logits),
        "max_diffs": [],
        "failed_positions": [],
        "statistics": {}
    }
    
    for pos in range(len(single_logits)):
        # Ensure both are evaluated and in float32
        single_f32 = single_logits[pos].astype(mx.float32)
        distributed_f32 = distributed_logits[pos].astype(mx.float32)
        mx.eval(single_f32)
        mx.eval(distributed_f32)
        
        # Convert to numpy for detailed analysis
        single_np = np.array(single_f32, dtype=np.float32)
        distributed_np = np.array(distributed_f32, dtype=np.float32)
        
        # Compute differences
        diff = np.abs(single_np - distributed_np)
        max_diff = np.max(diff)
        results["max_diffs"].append(max_diff)
        
        # Check tolerance
        if max_diff >= tolerance:
            results["passed"] = False
            results["failed_positions"].append(pos)
            
            # Find worst differences
            worst_indices = np.argsort(diff)[-10:][::-1]
            worst_diffs = [(int(idx), float(single_np[idx]), float(distributed_np[idx]), float(diff[idx])) 
                          for idx in worst_indices]
            
            logger.error(f"\n❌ Position {pos}: max|Δ| = {max_diff:.6f} EXCEEDS tolerance {tolerance}")
            logger.error("  Top 10 differences:")
            for idx, s_val, d_val, d in worst_diffs[:5]:
                logger.error(f"    Token {idx}: single={s_val:.6f}, distributed={d_val:.6f}, |Δ|={d:.6f}")
        else:
            logger.info(f"✅ Position {pos}: max|Δ| = {max_diff:.6f} < {tolerance}")
    
    # Compute statistics
    all_max_diffs = results["max_diffs"]
    results["statistics"] = {
        "max_diff_overall": float(max(all_max_diffs)),
        "mean_max_diff": float(np.mean(all_max_diffs)),
        "std_max_diff": float(np.std(all_max_diffs)),
        "positions_failed": len(results["failed_positions"]),
        "failure_rate": len(results["failed_positions"]) / len(single_logits)
    }
    
    return results

async def main():
    parser = argparse.ArgumentParser(description="Test logit consistency between single and distributed inference")
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-8bit", help="Model to test")
    parser.add_argument("--prompt", default="Once upon a time, there was a", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Maximum acceptable logit difference")
    parser.add_argument("--worker-port", type=int, default=50052, help="Port for worker 1")
    
    args = parser.parse_args()
    
    # Tokenize prompt (we'll use a simple tokenizer for consistency)
    from mlx_lm import load
    _, tokenizer = load(args.model)
    prompt_tokens = tokenizer.encode(args.prompt)
    
    logger.info(f"Test configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Prompt: '{args.prompt}'")
    logger.info(f"  Prompt tokens: {prompt_tokens}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    logger.info(f"  Tolerance: {args.tolerance}")
    
    try:
        # Wait a bit for worker to be ready
        logger.info("Waiting for worker to initialize...")
        import time
        time.sleep(3)
        
        # Run single-node inference
        logger.info("\n=== SINGLE-NODE INFERENCE ===")
        single_logits, single_tokens = await run_single_node_generation(args.model, prompt_tokens, args.max_tokens)
        
        # Run distributed inference
        logger.info("\n=== DISTRIBUTED INFERENCE ===")
        distributed_logits, distributed_tokens = await run_distributed_generation(args.model, prompt_tokens, args.max_tokens)
        
        # Compare results
        logger.info("\n=== COMPARING LOGITS ===")
        results = compare_logits_detailed(single_logits, distributed_logits, args.tolerance)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY REPORT")
        logger.info("="*60)
        logger.info(f"Positions compared: {results['positions']}")
        logger.info(f"Maximum difference: {results['statistics']['max_diff_overall']:.6f}")
        logger.info(f"Mean of max diffs: {results['statistics']['mean_max_diff']:.6f}")
        logger.info(f"Std of max diffs: {results['statistics']['std_max_diff']:.6f}")
        logger.info(f"Failed positions: {results['statistics']['positions_failed']} / {results['positions']}")
        
        # Check token generation consistency
        logger.info("\n=== TOKEN GENERATION ===")
        tokens_match = single_tokens == distributed_tokens
        logger.info(f"Tokens match: {tokens_match}")
        if not tokens_match:
            logger.warning("Token mismatch detected:")
            for i, (s, d) in enumerate(zip(single_tokens, distributed_tokens)):
                if s != d:
                    logger.warning(f"  Position {i}: single={s}, distributed={d}")
        
        if results["passed"] and tokens_match:
            logger.info("\n✅ SUCCESS: All logits within tolerance and tokens match!")
            sys.exit(0)
        else:
            logger.error("\n❌ FAILURE: Logits exceed tolerance or tokens don't match!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError during test: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())