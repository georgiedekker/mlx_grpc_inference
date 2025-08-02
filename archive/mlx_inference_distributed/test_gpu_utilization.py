#!/usr/bin/env python3
"""
Test GPU Utilization for Distributed MLX Inference
This script monitors GPU activity across all 3 devices during distributed inference
"""

import asyncio
import aiohttp
import time
import json
import subprocess
import logging
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.live import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUUtilizationTester:
    def __init__(self):
        self.console = Console()
        self.devices = {
            "mini1": {"host": "localhost", "port": 8100},
            "mini2": {"host": "mini2.local", "port": 8101},
            "master": {"host": "master.local", "port": 8102}
        }
        self.api_url = "http://localhost:8100"  # Coordinator
        
    async def get_gpu_metrics(self, device_name: str) -> Dict:
        """Get GPU metrics for a device"""
        try:
            if device_name == "mini1":
                # Local GPU check
                result = subprocess.run(
                    ["sudo", "powermetrics", "--samplers", "gpu_power", "-i", "1000", "-n", "1"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                
                if result.returncode == 0:
                    # Parse powermetrics output
                    output = result.stdout
                    gpu_active = "GPU Active" in output
                    return {
                        "device": device_name,
                        "gpu_active": gpu_active,
                        "raw_output": output[:200]  # First 200 chars
                    }
            else:
                # Remote check would require SSH
                # For now, return placeholder
                return {
                    "device": device_name,
                    "gpu_active": False,
                    "raw_output": "Remote monitoring not implemented"
                }
                
        except Exception as e:
            logger.error(f"Failed to get GPU metrics for {device_name}: {e}")
            return {
                "device": device_name,
                "gpu_active": False,
                "error": str(e)
            }
    
    async def test_inference(self, prompt: str, max_tokens: int = 50) -> Dict:
        """Send inference request to coordinator"""
        try:
            payload = {
                "model": "mlx-community/Qwen3-1.7B-8bit",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        duration = time.time() - start_time
                        
                        return {
                            "success": True,
                            "duration": duration,
                            "response": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                            "tokens": len(result.get("choices", [{}])[0].get("message", {}).get("content", "").split())
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            logger.error(f"Inference request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def check_cluster_health(self) -> Dict:
        """Check health of all devices in cluster"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check coordinator health
                async with session.get(
                    f"{self.api_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        return {
                            "healthy": True,
                            "coordinator": health_data
                        }
                    else:
                        return {"healthy": False, "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def monitor_and_test(self):
        """Main monitoring and testing loop"""
        self.console.print("[bold green]🚀 MLX Distributed GPU Utilization Test[/bold green]")
        self.console.print("=" * 50)
        
        # Check cluster health
        self.console.print("\n[yellow]Checking cluster health...[/yellow]")
        health = await self.check_cluster_health()
        
        if health.get("healthy"):
            self.console.print("[green]✅ Cluster is healthy![/green]")
            self.console.print(f"Coordinator: {health.get('coordinator', {})}")
        else:
            self.console.print(f"[red]❌ Cluster health check failed: {health.get('error')}[/red]")
            return
        
        # Test prompts
        test_prompts = [
            "Explain quantum computing in one sentence.",
            "What is machine learning?",
            "Tell me a short joke about distributed computing."
        ]
        
        self.console.print("\n[yellow]Running inference tests and monitoring GPU...[/yellow]")
        
        for i, prompt in enumerate(test_prompts, 1):
            self.console.print(f"\n[cyan]Test {i}/{len(test_prompts)}:[/cyan] {prompt}")
            
            # Start GPU monitoring tasks
            gpu_tasks = {
                device: asyncio.create_task(self.get_gpu_metrics(device))
                for device in self.devices
            }
            
            # Run inference
            inference_task = asyncio.create_task(self.test_inference(prompt))
            
            # Wait for both
            inference_result = await inference_task
            gpu_results = {
                device: await task
                for device, task in gpu_tasks.items()
            }
            
            # Display results
            if inference_result["success"]:
                self.console.print(f"[green]✅ Inference completed in {inference_result['duration']:.2f}s[/green]")
                self.console.print(f"Response: {inference_result['response'][:100]}...")
                self.console.print(f"Tokens generated: {inference_result['tokens']}")
            else:
                self.console.print(f"[red]❌ Inference failed: {inference_result['error']}[/red]")
            
            # Display GPU status
            table = Table(title="GPU Activity During Inference")
            table.add_column("Device", style="cyan")
            table.add_column("GPU Active", style="green")
            table.add_column("Details", style="yellow")
            
            for device, metrics in gpu_results.items():
                active = "✅" if metrics.get("gpu_active") else "❌"
                details = metrics.get("error", metrics.get("raw_output", "")[:50])
                table.add_row(device, active, details)
            
            self.console.print(table)
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        self.console.print("\n[green]✅ GPU utilization test completed![/green]")

async def main():
    """Main entry point"""
    tester = GPUUtilizationTester()
    await tester.monitor_and_test()

if __name__ == "__main__":
    asyncio.run(main())