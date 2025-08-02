#!/usr/bin/env python3
"""
GPU Activity Monitor for Distributed MLX Inference
Monitors GPU utilization across all devices and validates distributed processing
"""

import asyncio
import subprocess
import json
import time
import logging
from typing import Dict, List, Optional
import aiohttp
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitor GPU activity across distributed devices"""
    
    def __init__(self):
        self.console = Console()
        self.devices = {
            "mini1": {"host": "mini1.local", "port": 8100, "grpc_port": 9100},
            "mini2": {"host": "mini2.local", "port": 8101, "grpc_port": 9101},
            "master": {"host": "master.local", "port": 8102, "grpc_port": 9102}
        }
        self.metrics_history = {device: [] for device in self.devices}
        
    async def get_local_gpu_metrics(self) -> Dict[str, float]:
        """Get local GPU metrics using asitop or system commands"""
        try:
            # Try asitop first
            result = subprocess.run(
                ["asitop", "--dump", "1"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse asitop output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "GPU" in line and "%" in line:
                        # Extract GPU percentage
                        gpu_percent = float(line.split('%')[0].split()[-1])
                        return {"gpu_percent": gpu_percent}
            
            # Fallback to ioreg for Apple Silicon
            result = subprocess.run(
                ["ioreg", "-l", "-w0", "-r", "-c", "AGXAccelerator"],
                capture_output=True,
                text=True
            )
            
            # Parse ioreg output (simplified)
            if "Device Utilization" in result.stdout:
                # Extract utilization (this is a simplified example)
                return {"gpu_percent": 0.0}  # Would need proper parsing
                
        except Exception as e:
            logger.error(f"Failed to get local GPU metrics: {e}")
            
        return {"gpu_percent": 0.0}
    
    async def get_device_stats(self, device_name: str) -> Optional[Dict]:
        """Get stats from a specific device"""
        device_info = self.devices[device_name]
        url = f"http://{device_info['host']}:{device_info['port']}/device-info"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Get GPU metrics for this device
                        if device_name == "mini1":
                            gpu_metrics = await self.get_local_gpu_metrics()
                        else:
                            # Would need SSH to get remote GPU metrics
                            gpu_metrics = {"gpu_percent": 0.0}
                        
                        return {
                            "status": "online",
                            "model_loaded": data.get("model_loaded", False),
                            "distributed_ready": data.get("distributed_engine_ready", False),
                            "layers": data.get("layers", []),
                            "gpu_percent": gpu_metrics.get("gpu_percent", 0.0),
                            "distributed_stats": data.get("distributed_stats", {})
                        }
        except Exception as e:
            logger.debug(f"Failed to get stats from {device_name}: {e}")
            
        return {
            "status": "offline",
            "model_loaded": False,
            "distributed_ready": False,
            "layers": [],
            "gpu_percent": 0.0,
            "distributed_stats": {}
        }
    
    async def test_distributed_inference(self) -> Dict:
        """Send a test inference request and monitor GPU activity"""
        url = f"http://{self.devices['mini1']['host']}:{self.devices['mini1']['port']}/v1/chat/completions"
        
        test_payload = {
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=test_payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        latency = time.time() - start_time
                        
                        return {
                            "success": True,
                            "latency": latency,
                            "response": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                            "tokens": len(result.get("choices", [{}])[0].get("message", {}).get("content", "").split())
                        }
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            
        return {"success": False, "latency": 0, "response": "", "tokens": 0}
    
    def create_dashboard(self, stats: Dict) -> Layout:
        """Create a rich dashboard layout"""
        layout = Layout()
        
        # Create device status table
        device_table = Table(title="Device Status", expand=True)
        device_table.add_column("Device", style="cyan")
        device_table.add_column("Status", style="green")
        device_table.add_column("GPU %", style="yellow")
        device_table.add_column("Layers", style="blue")
        device_table.add_column("Model", style="magenta")
        device_table.add_column("Distributed", style="red")
        
        for device_name, device_stats in stats["devices"].items():
            status_emoji = "🟢" if device_stats["status"] == "online" else "🔴"
            gpu_bar = self.create_progress_bar(device_stats["gpu_percent"])
            layers_str = f"{min(device_stats['layers'])}-{max(device_stats['layers'])}" if device_stats['layers'] else "N/A"
            model_emoji = "✅" if device_stats["model_loaded"] else "❌"
            dist_emoji = "✅" if device_stats["distributed_ready"] else "❌"
            
            device_table.add_row(
                device_name,
                f"{status_emoji} {device_stats['status']}",
                f"{gpu_bar} {device_stats['gpu_percent']:.1f}%",
                layers_str,
                model_emoji,
                dist_emoji
            )
        
        # Create performance metrics panel
        perf_text = f"""
[bold]Inference Performance[/bold]
━━━━━━━━━━━━━━━━━━━━━
Total Requests: {stats['total_requests']}
Success Rate: {stats['success_rate']:.1f}%
Avg Latency: {stats['avg_latency']:.2f}s
Tokens/sec: {stats['tokens_per_sec']:.1f}

[bold]Last Test Result[/bold]
━━━━━━━━━━━━━━━━━━━━━
Success: {'✅' if stats['last_test']['success'] else '❌'}
Latency: {stats['last_test']['latency']:.2f}s
Tokens: {stats['last_test']['tokens']}
"""
        
        # Create GPU activity history
        gpu_history = self.create_gpu_history_chart()
        
        # Arrange layout
        layout.split_column(
            Layout(Panel(device_table, title="🖥️  Cluster Status"), name="devices"),
            Layout(name="bottom")
        )
        
        layout["bottom"].split_row(
            Layout(Panel(perf_text, title="📊 Performance"), name="performance"),
            Layout(Panel(gpu_history, title="📈 GPU Activity History"), name="history")
        )
        
        return layout
    
    def create_progress_bar(self, percent: float) -> str:
        """Create a simple progress bar"""
        filled = int(percent / 10)
        empty = 10 - filled
        return f"[{'█' * filled}{'░' * empty}]"
    
    def create_gpu_history_chart(self) -> str:
        """Create a simple ASCII chart of GPU history"""
        chart_lines = []
        
        for device, history in self.metrics_history.items():
            if history:
                recent = history[-20:]  # Last 20 samples
                sparkline = self.create_sparkline([h["gpu_percent"] for h in recent])
                chart_lines.append(f"{device}: {sparkline}")
        
        return "\n".join(chart_lines) if chart_lines else "No history data yet"
    
    def create_sparkline(self, data: List[float]) -> str:
        """Create a sparkline chart"""
        if not data:
            return ""
        
        chars = "▁▂▃▄▅▆▇█"
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return chars[0] * len(data)
        
        sparkline = ""
        for value in data:
            idx = int((value - min_val) / (max_val - min_val) * (len(chars) - 1))
            sparkline += chars[idx]
        
        return sparkline
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        total_requests = 0
        successful_requests = 0
        total_latency = 0
        last_test = {"success": False, "latency": 0, "response": "", "tokens": 0}
        
        with Live(self.create_dashboard({
            "devices": {},
            "total_requests": 0,
            "success_rate": 0,
            "avg_latency": 0,
            "tokens_per_sec": 0,
            "last_test": last_test
        }), refresh_per_second=1) as live:
            
            while True:
                try:
                    # Collect device stats
                    device_stats = {}
                    for device_name in self.devices:
                        stats = await self.get_device_stats(device_name)
                        device_stats[device_name] = stats
                        
                        # Store history
                        self.metrics_history[device_name].append({
                            "timestamp": time.time(),
                            "gpu_percent": stats["gpu_percent"]
                        })
                        
                        # Keep only last 100 samples
                        if len(self.metrics_history[device_name]) > 100:
                            self.metrics_history[device_name].pop(0)
                    
                    # Run inference test every 10 seconds
                    if total_requests == 0 or (total_requests % 10 == 0):
                        test_result = await self.test_distributed_inference()
                        total_requests += 1
                        
                        if test_result["success"]:
                            successful_requests += 1
                            total_latency += test_result["latency"]
                            last_test = test_result
                    
                    # Calculate metrics
                    success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
                    avg_latency = (total_latency / successful_requests) if successful_requests > 0 else 0
                    tokens_per_sec = (last_test["tokens"] / last_test["latency"]) if last_test["latency"] > 0 else 0
                    
                    # Update dashboard
                    stats = {
                        "devices": device_stats,
                        "total_requests": total_requests,
                        "success_rate": success_rate,
                        "avg_latency": avg_latency,
                        "tokens_per_sec": tokens_per_sec,
                        "last_test": last_test
                    }
                    
                    live.update(self.create_dashboard(stats))
                    
                except Exception as e:
                    logger.error(f"Monitor loop error: {e}")
                
                await asyncio.sleep(1)

async def main():
    """Main entry point"""
    monitor = GPUMonitor()
    
    print("🚀 Starting GPU Activity Monitor for Distributed MLX Inference")
    print("Press Ctrl+C to stop\n")
    
    try:
        await monitor.monitor_loop()
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")

if __name__ == "__main__":
    asyncio.run(main())