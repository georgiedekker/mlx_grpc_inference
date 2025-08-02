#!/usr/bin/env python3
"""
Monitor GPU activity across all devices in the cluster.
"""

import asyncio
import subprocess
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from datetime import datetime
import psutil

console = Console()


def get_local_gpu_activity():
    """Get local GPU activity using powermetrics."""
    try:
        # Run powermetrics for 1 second
        result = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "gpu_power", "-i", "1000", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            output = result.stdout
            # Look for GPU activity indicators
            gpu_active = "GPU Active" in output or "GPU Power" in output
            
            # Extract GPU frequency if available
            gpu_freq = "Unknown"
            for line in output.split('\n'):
                if "GPU Frequency" in line:
                    gpu_freq = line.split(':')[-1].strip()
                    break
            
            return {
                "active": gpu_active,
                "frequency": gpu_freq,
                "status": "Active" if gpu_active else "Idle"
            }
    except Exception as e:
        return {
            "active": False,
            "frequency": "Error",
            "status": f"Error: {str(e)}"
        }
    
    return {
        "active": False,
        "frequency": "Unknown",
        "status": "No data"
    }


def get_system_metrics():
    """Get system metrics."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_total_gb": memory.total / (1024**3)
    }


def create_monitoring_table():
    """Create the monitoring table."""
    table = Table(title=f"MLX GPU Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    table.add_column("Device", style="cyan", width=12)
    table.add_column("GPU Status", style="green", width=15)
    table.add_column("GPU Frequency", style="yellow", width=20)
    table.add_column("CPU %", style="magenta", width=10)
    table.add_column("Memory", style="blue", width=20)
    
    # Local device (mini1)
    gpu_info = get_local_gpu_activity()
    sys_metrics = get_system_metrics()
    
    table.add_row(
        "mini1 (local)",
        gpu_info["status"],
        gpu_info["frequency"],
        f"{sys_metrics['cpu_percent']:.1f}%",
        f"{sys_metrics['memory_used_gb']:.1f}/{sys_metrics['memory_total_gb']:.1f} GB ({sys_metrics['memory_percent']:.1f}%)"
    )
    
    # Remote devices (simplified - would need SSH in real implementation)
    table.add_row(
        "mini2",
        "Remote",
        "N/A",
        "N/A",
        "N/A"
    )
    
    table.add_row(
        "master",
        "Remote",
        "N/A",
        "N/A",
        "N/A"
    )
    
    return table


async def monitor_loop():
    """Main monitoring loop."""
    console.print("[bold green]🖥️  MLX GPU Activity Monitor[/bold green]")
    console.print("Press Ctrl+C to stop\n")
    
    with Live(create_monitoring_table(), refresh_per_second=1) as live:
        while True:
            live.update(create_monitoring_table())
            await asyncio.sleep(2)


async def main():
    """Main entry point."""
    try:
        await monitor_loop()
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())