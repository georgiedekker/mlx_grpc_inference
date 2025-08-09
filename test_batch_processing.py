#!/usr/bin/env python3
"""
Test script for batch processing with the tensor parallel system.
"""
import asyncio
import aiohttp
import time
import json
import logging
from typing import List, Dict, Any
import statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


async def send_request(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int = 50,
    request_id: int = 0
) -> Dict[str, Any]:
    """Send a single request to the API."""
    url = "http://localhost:8100/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                elapsed = time.time() - start_time
                
                return {
                    "request_id": request_id,
                    "success": True,
                    "response": result['choices'][0]['message']['content'],
                    "tokens": result['usage']['total_tokens'],
                    "prompt_tokens": result['usage']['prompt_tokens'],
                    "completion_tokens": result['usage']['completion_tokens'],
                    "eval_speed": result['usage'].get('eval_tokens_per_second', 0),
                    "latency": elapsed,
                    "error": None
                }
            else:
                return {
                    "request_id": request_id,
                    "success": False,
                    "latency": elapsed,
                    "error": f"HTTP {response.status}"
                }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "request_id": request_id,
            "success": False,
            "latency": elapsed,
            "error": str(e)
        }


async def test_concurrent_requests(num_requests: int = 5):
    """Test multiple concurrent requests."""
    logger.info(f"Testing {num_requests} concurrent requests...")
    
    prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "What is the speed of light?",
        "Define artificial intelligence",
        "What is tensor parallelism?",
        "Explain distributed computing",
        "What is GPU acceleration?",
        "Define neural networks",
        "What is deep learning?",
        "Explain transformers"
    ]
    
    # Use different prompts, cycling if needed
    test_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Send all requests concurrently
        tasks = [
            send_request(session, prompt, max_tokens=30, request_id=i)
            for i, prompt in enumerate(test_prompts)
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Concurrent Test Results ({num_requests} requests)")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Successful: {len(successful)}/{num_requests}")
    logger.info(f"Failed: {len(failed)}/{num_requests}")
    
    if successful:
        latencies = [r['latency'] for r in successful]
        tokens = [r['tokens'] for r in successful]
        eval_speeds = [r['eval_speed'] for r in successful if r['eval_speed'] > 0]
        
        logger.info(f"\nLatency Statistics:")
        logger.info(f"  Mean: {statistics.mean(latencies):.2f}s")
        logger.info(f"  Median: {statistics.median(latencies):.2f}s")
        logger.info(f"  Min: {min(latencies):.2f}s")
        logger.info(f"  Max: {max(latencies):.2f}s")
        
        logger.info(f"\nThroughput:")
        logger.info(f"  Requests/second: {len(successful)/total_time:.2f}")
        logger.info(f"  Total tokens: {sum(tokens)}")
        logger.info(f"  Tokens/second: {sum(tokens)/total_time:.2f}")
        
        if eval_speeds:
            logger.info(f"\nGeneration Speed:")
            logger.info(f"  Mean: {statistics.mean(eval_speeds):.1f} tok/s")
            logger.info(f"  Median: {statistics.median(eval_speeds):.1f} tok/s")
    
    if failed:
        logger.error(f"\nFailed Requests:")
        for r in failed:
            logger.error(f"  Request {r['request_id']}: {r['error']}")
    
    return results


async def test_sequential_requests(num_requests: int = 5):
    """Test sequential requests for comparison."""
    logger.info(f"\nTesting {num_requests} sequential requests...")
    
    prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "What is the speed of light?",
        "Define artificial intelligence",
        "What is tensor parallelism?"
    ]
    
    test_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        results = []
        
        # Send requests one by one
        for i, prompt in enumerate(test_prompts):
            result = await send_request(session, prompt, max_tokens=30, request_id=i)
            results.append(result)
        
        total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r['success']]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Sequential Test Results ({num_requests} requests)")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Successful: {len(successful)}/{num_requests}")
    
    if successful:
        latencies = [r['latency'] for r in successful]
        logger.info(f"\nLatency Statistics:")
        logger.info(f"  Mean: {statistics.mean(latencies):.2f}s")
        logger.info(f"  Total: {sum(latencies):.2f}s")
        logger.info(f"\nThroughput:")
        logger.info(f"  Requests/second: {len(successful)/total_time:.2f}")
    
    return results


async def stress_test(duration: int = 30, requests_per_second: int = 2):
    """
    Stress test the system with continuous load.
    
    Args:
        duration: Test duration in seconds
        requests_per_second: Target request rate
    """
    logger.info(f"\nStarting stress test: {duration}s at {requests_per_second} req/s")
    
    prompts = [
        "Explain this concept briefly:",
        "What is the definition of",
        "Give me a short summary of",
        "Describe in simple terms:"
    ]
    
    topics = [
        "machine learning",
        "neural networks",
        "quantum computing",
        "blockchain",
        "cloud computing",
        "edge computing",
        "5G networks",
        "IoT devices"
    ]
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        request_interval = 1.0 / requests_per_second
        results = []
        request_id = 0
        
        while (time.time() - start_time) < duration:
            # Generate random prompt
            prompt = f"{prompts[request_id % len(prompts)]} {topics[request_id % len(topics)]}"
            
            # Send request without waiting
            task = asyncio.create_task(
                send_request(session, prompt, max_tokens=20, request_id=request_id)
            )
            results.append(task)
            request_id += 1
            
            # Wait for next request time
            await asyncio.sleep(request_interval)
        
        # Wait for all pending requests to complete
        logger.info("Waiting for pending requests to complete...")
        completed_results = await asyncio.gather(*results)
    
    # Analyze results
    successful = [r for r in completed_results if r['success']]
    failed = [r for r in completed_results if not r['success']]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Stress Test Results")
    logger.info(f"{'='*60}")
    logger.info(f"Duration: {duration}s")
    logger.info(f"Total requests: {len(completed_results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Success rate: {len(successful)/len(completed_results)*100:.1f}%")
    
    if successful:
        latencies = [r['latency'] for r in successful]
        logger.info(f"\nLatency under load:")
        logger.info(f"  P50: {statistics.quantiles(latencies, n=2)[0]:.2f}s")
        logger.info(f"  P90: {statistics.quantiles(latencies, n=10)[8]:.2f}s")
        logger.info(f"  P99: {statistics.quantiles(latencies, n=100)[98]:.2f}s")
        logger.info(f"  Max: {max(latencies):.2f}s")


async def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("MLX Tensor Parallel Batch Processing Test")
    logger.info("=" * 80)
    
    # Check if API is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8100/health") as response:
                if response.status != 200:
                    logger.error("API is not running! Start with: ./launch_tensor_parallel.sh")
                    return
                health = await response.json()
                logger.info(f"API Status: {health}")
    except Exception as e:
        logger.error(f"Cannot connect to API: {e}")
        logger.error("Start the system with: ./launch_tensor_parallel.sh")
        return
    
    # Run tests
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: Sequential Baseline")
    logger.info("="*80)
    await test_sequential_requests(5)
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: Concurrent Requests")
    logger.info("="*80)
    await test_concurrent_requests(5)
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: Higher Concurrency")
    logger.info("="*80)
    await test_concurrent_requests(10)
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: Stress Test")
    logger.info("="*80)
    await stress_test(duration=20, requests_per_second=2)
    
    logger.info("\n" + "="*80)
    logger.info("Test Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())