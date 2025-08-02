#!/usr/bin/env python3
"""
Test script for distributed MLX inference.
"""

import asyncio
import aiohttp
import json
import time
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


async def test_health(session: aiohttp.ClientSession, base_url: str):
    """Test health endpoint."""
    try:
        async with session.get(f"{base_url}/health") as response:
            if response.status == 200:
                data = await response.json()
                return True, data
            else:
                return False, f"HTTP {response.status}"
    except Exception as e:
        return False, str(e)


async def test_cluster_status(session: aiohttp.ClientSession, base_url: str):
    """Test cluster status endpoint."""
    try:
        async with session.get(f"{base_url}/cluster/status") as response:
            if response.status == 200:
                data = await response.json()
                return True, data
            else:
                return False, f"HTTP {response.status}"
    except Exception as e:
        return False, str(e)


async def test_chat_completion(session: aiohttp.ClientSession, base_url: str, prompt: str):
    """Test chat completion endpoint."""
    payload = {
        "model": "mlx-community/Qwen3-1.7B-8bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        start_time = time.time()
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 200:
                data = await response.json()
                elapsed = time.time() - start_time
                return True, data, elapsed
            else:
                error = await response.text()
                return False, f"HTTP {response.status}: {error}", 0
    except Exception as e:
        return False, str(e), 0


async def main():
    """Run tests."""
    base_url = "http://localhost:8100"
    
    console.print("[bold blue]🧪 Testing MLX Distributed Inference Cluster[/bold blue]")
    console.print(f"API URL: {base_url}\n")
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Health check
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Testing health endpoint...", total=None)
            success, data = await test_health(session, base_url)
            progress.remove_task(task)
        
        if success:
            console.print("✅ Health Check: [green]PASSED[/green]")
            table = Table(title="Health Status")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="yellow")
            for key, value in data.items():
                table.add_row(str(key), str(value))
            console.print(table)
        else:
            console.print(f"❌ Health Check: [red]FAILED[/red] - {data}")
            sys.exit(1)
        
        console.print()
        
        # Test 2: Cluster status
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Testing cluster status...", total=None)
            success, data = await test_cluster_status(session, base_url)
            progress.remove_task(task)
        
        if success:
            console.print("✅ Cluster Status: [green]PASSED[/green]")
            
            # Coordinator info
            console.print("\n[bold]Coordinator:[/bold]")
            coord_table = Table()
            coord_table.add_column("Property", style="cyan")
            coord_table.add_column("Value", style="yellow")
            for key, value in data.get('coordinator', {}).items():
                coord_table.add_row(str(key), str(value))
            console.print(coord_table)
            
            # Workers info
            console.print("\n[bold]Workers:[/bold]")
            workers_table = Table()
            workers_table.add_column("Device ID", style="cyan")
            workers_table.add_column("Status", style="yellow")
            workers_table.add_column("Layers", style="green")
            workers_table.add_column("GPU %", style="magenta")
            
            for worker in data.get('workers', []):
                workers_table.add_row(
                    worker.get('device_id', 'unknown'),
                    worker.get('status', 'unknown'),
                    str(worker.get('assigned_layers', [])),
                    f"{worker.get('gpu_utilization', 0):.1f}%"
                )
            console.print(workers_table)
        else:
            console.print(f"❌ Cluster Status: [red]FAILED[/red] - {data}")
        
        console.print()
        
        # Test 3: Chat completions
        test_prompts = [
            "What is machine learning?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about distributed computing."
        ]
        
        console.print("[bold]Testing Chat Completions:[/bold]")
        results_table = Table()
        results_table.add_column("Prompt", style="cyan", width=40)
        results_table.add_column("Response", style="yellow", width=50)
        results_table.add_column("Time", style="green")
        results_table.add_column("Tokens", style="magenta")
        
        for prompt in test_prompts:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Testing: {prompt[:30]}...", total=None)
                success, data, elapsed = await test_chat_completion(session, base_url, prompt)
                progress.remove_task(task)
            
            if success:
                response = data['choices'][0]['message']['content']
                tokens = data['usage']['completion_tokens']
                results_table.add_row(
                    prompt[:40] + "..." if len(prompt) > 40 else prompt,
                    response[:50] + "..." if len(response) > 50 else response,
                    f"{elapsed:.2f}s",
                    str(tokens)
                )
            else:
                results_table.add_row(
                    prompt[:40] + "..." if len(prompt) > 40 else prompt,
                    f"[red]ERROR: {data}[/red]",
                    "-",
                    "-"
                )
        
        console.print(results_table)
    
    console.print("\n[bold green]✅ All tests completed![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())