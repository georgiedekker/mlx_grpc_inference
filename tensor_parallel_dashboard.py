#!/usr/bin/env python3
"""
Performance monitoring dashboard for tensor parallel inference.
"""
import asyncio
import sys
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import time
import psutil
import mlx.core as mx

# Add project root to path
sys.path.append('/Users/mini1/Movies/mlx_grpc_inference')

from src.performance_monitor import get_monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for dashboard
app = FastAPI(title="MLX Tensor Parallel Dashboard")

# Get monitor instance
monitor = get_monitor()

@app.get("/")
async def dashboard():
    """Serve dashboard HTML."""
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>MLX Tensor Parallel Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1e1e1e;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        h1 {
            margin: 0;
            font-size: 28px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #3a3a3a;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #4ade80;
        }
        .metric-label {
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            margin-top: 5px;
        }
        .chart-container {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #3a3a3a;
            margin-bottom: 20px;
            height: 300px;
        }
        .status {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-healthy { background: #4ade80; color: #000; }
        .status-warning { background: #fbbf24; color: #000; }
        .status-error { background: #f87171; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ MLX Tensor Parallel Inference Dashboard</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">
                Monitoring tensor-parallel execution across mini1 and mini2
            </p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="api-status">
                    <span class="status status-warning">LOADING</span>
                </div>
                <div class="metric-label">API Status</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="tokens-per-second">0</div>
                <div class="metric-label">Tokens/Second</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="latency">0</div>
                <div class="metric-label">Latency (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="gpu-memory">0</div>
                <div class="metric-label">GPU Memory (MB)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="active-sessions">0</div>
                <div class="metric-label">Active Sessions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="total-requests">0</div>
                <div class="metric-label">Total Requests</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="performance-chart"></canvas>
        </div>
        
        <div style="background: #2a2a2a; padding: 20px; border-radius: 8px; border: 1px solid #3a3a3a;">
            <h3>Test Commands</h3>
            <pre style="color: #4ade80; font-family: 'Courier New', monospace;">
# Test the API
curl -X POST "http://localhost:8100/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is tensor parallelism?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Check health
curl http://localhost:8100/health</pre>
        </div>
    </div>
    
    <script>
        // Initialize performance chart
        const ctx = document.getElementById('performance-chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Tokens/Second',
                    data: [],
                    borderColor: '#4ade80',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#fff' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#999' },
                        grid: { color: '#3a3a3a' }
                    },
                    y: {
                        ticks: { color: '#999' },
                        grid: { color: '#3a3a3a' }
                    }
                }
            }
        });
        
        let totalRequests = 0;
        
        // Check API health
        async function checkApiHealth() {
            try {
                const response = await fetch('http://localhost:8100/health');
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('api-status').innerHTML = 
                        '<span class="status status-healthy">ONLINE</span>';
                } else {
                    document.getElementById('api-status').innerHTML = 
                        '<span class="status status-error">OFFLINE</span>';
                }
            } catch (error) {
                document.getElementById('api-status').innerHTML = 
                    '<span class="status status-error">OFFLINE</span>';
            }
        }
        
        // Update metrics
        async function updateMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                
                if (data.current) {
                    document.getElementById('tokens-per-second').textContent = 
                        data.current.tokens_per_second.toFixed(1);
                    document.getElementById('latency').textContent = 
                        data.current.latency_ms.toFixed(1);
                    document.getElementById('gpu-memory').textContent = 
                        data.current.gpu_memory_mb.toFixed(0);
                    document.getElementById('active-sessions').textContent = 
                        data.current.active_sessions;
                    
                    // Update chart
                    const time = new Date().toLocaleTimeString();
                    chart.data.labels.push(time);
                    chart.data.datasets[0].data.push(data.current.tokens_per_second);
                    
                    // Keep only last 30 points
                    if (chart.data.labels.length > 30) {
                        chart.data.labels.shift();
                        chart.data.datasets[0].data.shift();
                    }
                    
                    chart.update();
                }
                
                if (data.totals) {
                    totalRequests = data.totals.total_tokens_processed || totalRequests;
                    document.getElementById('total-requests').textContent = totalRequests;
                }
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
            }
        }
        
        // Start updating
        checkApiHealth();
        setInterval(checkApiHealth, 5000);
        setInterval(updateMetrics, 1000);
        updateMetrics();
    </script>
</body>
</html>
    '''
    return HTMLResponse(content=html_content)

@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics as JSON."""
    return JSONResponse(monitor.get_metrics_summary())

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "dashboard": "running"}

if __name__ == "__main__":
    logger.info(f"Starting Tensor Parallel Dashboard on port 8888")
    logger.info(f"Dashboard URL: http://localhost:8888")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="error")
