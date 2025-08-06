#!/usr/bin/env python3
"""
Standalone monitoring script that watches the tensor parallel system.
Doesn't interfere with the main system - just observes and reports.
"""
import asyncio
import time
import psutil
import mlx.core as mx
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


async def monitor_system():
    """Monitor the running tensor parallel system."""
    logger.info("Starting system monitoring...")
    logger.info("This monitors the existing tensor parallel system without interfering")
    
    while True:
        try:
            # Check if API is running
            import requests
            try:
                response = requests.get("http://localhost:8100/health", timeout=1)
                if response.ok:
                    health = response.json()
                    status = "🟢 ONLINE"
                else:
                    status = "🔴 OFFLINE"
            except:
                status = "🔴 OFFLINE"
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_used_gb = (memory.total - memory.available) / (1024**3)
            
            # Get GPU memory if available
            try:
                gpu_info = mx.metal.get_memory_info()
                gpu_memory_mb = gpu_info.get('current', 0) / (1024 * 1024)
            except:
                gpu_memory_mb = 0
            
            # Get network stats
            net_io = psutil.net_io_counters()
            
            # Print status line
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"API: {status} | "
                  f"CPU: {cpu_percent:.1f}% | "
                  f"RAM: {memory_used_gb:.1f}GB | "
                  f"GPU: {gpu_memory_mb:.0f}MB | "
                  f"Net: ↑{net_io.bytes_sent/(1024**2):.0f}MB ↓{net_io.bytes_recv/(1024**2):.0f}MB", 
                  end='', flush=True)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        
        await asyncio.sleep(2)
    
    print("\nMonitoring stopped")


async def test_api():
    """Test the API with a simple request."""
    import aiohttp
    import json
    
    logger.info("Testing API endpoint...")
    
    url = "http://localhost:8100/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": "Say hello in 5 words"}],
        "max_tokens": 10,
        "temperature": 0.7
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("✅ API test successful!")
                    logger.info(f"Response: {result['choices'][0]['message']['content']}")
                    if 'usage' in result:
                        usage = result['usage']
                        logger.info(f"Tokens: {usage.get('total_tokens', 'N/A')}")
                        logger.info(f"Speed: {usage.get('eval_tokens_per_second', 'N/A')} tok/s")
                else:
                    logger.error(f"API test failed: {response.status}")
    except Exception as e:
        logger.error(f"API test error: {e}")


async def main():
    """Main monitoring function."""
    print("=" * 60)
    print("MLX Tensor Parallel System Monitor")
    print("=" * 60)
    print()
    
    # Test the API
    await test_api()
    print()
    
    # Start monitoring
    print("Starting real-time monitoring (Ctrl+C to stop)...")
    print("-" * 60)
    await monitor_system()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")