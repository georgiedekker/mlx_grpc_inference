"""
Monitoring dashboard for distributed MLX inference system.

This module provides a web-based dashboard for monitoring:
- Real-time device metrics and health status
- Inference throughput and latency
- Distributed system topology
- Historical trends and alerts
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from .metrics import MetricsCollector
from .health import HealthMonitor, HealthStatus

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Web-based monitoring dashboard for the distributed system."""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 health_monitor: HealthMonitor,
                 host: str = "0.0.0.0",
                 port: int = 8080):
        """
        Initialize monitoring dashboard.
        
        Args:
            metrics_collector: Metrics collector instance
            health_monitor: Health monitor instance
            host: Dashboard host address
            port: Dashboard port
        """
        self.metrics_collector = metrics_collector
        self.health_monitor = health_monitor
        self.host = host
        self.port = port
        
        # FastAPI app
        self.app = FastAPI(
            title="MLX Distributed Inference Monitor",
            description="Real-time monitoring dashboard for distributed MLX inference",
            version="1.0.0"
        )
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        # Background task for broadcasting updates
        self._broadcast_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized monitoring dashboard at {host}:{port}")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return HTMLResponse(content=self._get_dashboard_html(), status_code=200)
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics."""
            try:
                return JSONResponse(content=self.metrics_collector.export_metrics())
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/health")
        async def get_health():
            """Get health status."""
            try:
                return JSONResponse(content=self.health_monitor.get_health_summary())
            except Exception as e:
                logger.error(f"Error getting health: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/devices")
        async def get_devices():
            """Get information about all devices."""
            try:
                device_metrics = self.metrics_collector.get_device_metrics()
                device_health = self.health_monitor.get_device_health()
                
                devices = {}
                
                # Combine metrics and health data
                all_device_ids = set(device_metrics.keys()) | set(device_health.keys())
                
                for device_id in all_device_ids:
                    metrics = device_metrics.get(device_id)
                    health = device_health.get(device_id)
                    
                    devices[device_id] = {
                        'device_id': device_id,
                        'metrics': {
                            'cpu_percent': metrics.cpu_percent if metrics else 0,
                            'memory_percent': metrics.memory_percent if metrics else 0,
                            'memory_used_gb': metrics.memory_used_gb if metrics else 0,
                            'memory_total_gb': metrics.memory_total_gb if metrics else 0,
                            'gpu_utilization': metrics.gpu_utilization if metrics else None,
                            'model_loaded': metrics.model_loaded if metrics else False,
                            'assigned_layers': metrics.assigned_layers if metrics else [],
                            'timestamp': metrics.timestamp if metrics else 0
                        },
                        'health': {
                            'status': health.status.value if health else 'unknown',
                            'is_responsive': health.is_responsive if health else False,
                            'last_seen': health.last_seen if health else 0,
                            'checks_count': len(health.checks) if health else 0
                        }
                    }
                
                return JSONResponse(content=devices)
                
            except Exception as e:
                logger.error(f"Error getting devices: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/throughput")
        async def get_throughput():
            """Get throughput metrics."""
            try:
                return JSONResponse(content=self.metrics_collector.get_throughput_metrics())
            except Exception as e:
                logger.error(f"Error getting throughput: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/stats")
        async def get_stats():
            """Get inference statistics."""
            try:
                stats = self.metrics_collector.get_inference_stats(time_window_seconds=300)  # Last 5 minutes
                return JSONResponse(content=stats)
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                # Keep connection alive and handle disconnection
                while True:
                    await websocket.receive_text()  # Wait for client messages
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
                logger.info("WebSocket client disconnected")
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLX Distributed Inference Monitor</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .header h1 {
            margin: 0;
            color: #333;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .card h3 {
            margin: 0 0 15px 0;
            color: #333;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 500;
            color: #666;
        }
        
        .metric-value {
            font-weight: bold;
        }
        
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .status-unknown { color: #6c757d; }
        
        .device-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .device-card {
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid #28a745;
        }
        
        .device-card.warning {
            border-left-color: #ffc107;
        }
        
        .device-card.critical {
            border-left-color: #dc3545;
        }
        
        .device-card.unknown {
            border-left-color: #6c757d;
        }
        
        .device-header {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        
        .progress-bar {
            width: 100%;
            height: 6px;
            background-color: #e9ecef;
            border-radius: 3px;
            overflow: hidden;
            margin: 5px 0;
        }
        
        .progress-fill {
            height: 100%;
            background-color: #007bff;
            transition: width 0.3s ease;
        }
        
        .progress-fill.warning { background-color: #ffc107; }
        .progress-fill.critical { background-color: #dc3545; }
        
        .update-time {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }
        
        .stats-row {
            display: flex;
            justify-content: space-around;
            text-align: center;
        }
        
        .stat-item {
            flex: 1;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>MLX Distributed Inference Monitor</h1>
        <p>Real-time monitoring of distributed inference system</p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>System Overview</h3>
            <div id="system-overview">
                <div class="metric">
                    <span class="metric-label">Total Devices</span>
                    <span class="metric-value" id="total-devices">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Healthy Devices</span>
                    <span class="metric-value" id="healthy-devices">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Overall Status</span>
                    <span class="metric-value" id="overall-status">-</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>Performance Metrics</h3>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-value" id="requests-per-sec">0</div>
                    <div class="stat-label">Req/sec</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="tokens-per-sec">0</div>
                    <div class="stat-label">Tokens/sec</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="avg-latency">0</div>
                    <div class="stat-label">Avg Latency (ms)</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>Request Statistics</h3>
            <div id="request-stats">
                <div class="metric">
                    <span class="metric-label">Total Requests</span>
                    <span class="metric-value" id="total-requests">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate</span>
                    <span class="metric-value" id="success-rate">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Tokens Generated</span>
                    <span class="metric-value" id="tokens-generated">-</span>
                </div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h3>Device Status</h3>
        <div id="devices-container" class="device-grid">
            <!-- Devices will be populated here -->
        </div>
    </div>
    
    <div class="update-time">
        Last updated: <span id="last-update">-</span>
    </div>
    
    <script>
        let ws;
        let reconnectInterval = 5000;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                fetchData(); // Fetch initial data
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected, reconnecting...');
                setTimeout(connectWebSocket, reconnectInterval);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        async function fetchData() {
            try {
                const [metrics, health, devices, throughput, stats] = await Promise.all([
                    fetch('/api/metrics').then(r => r.json()),
                    fetch('/api/health').then(r => r.json()),
                    fetch('/api/devices').then(r => r.json()),
                    fetch('/api/throughput').then(r => r.json()),
                    fetch('/api/stats').then(r => r.json())
                ]);
                
                updateDashboard({ metrics, health, devices, throughput, stats });
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        function updateDashboard(data) {
            if (data.devices) updateDevices(data.devices);
            if (data.throughput) updateThroughput(data.throughput);
            if (data.stats) updateStats(data.stats);
            if (data.health) updateSystemOverview(data.devices, data.health);
            
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }
        
        function updateSystemOverview(devices, health) {
            if (!devices) return;
            
            const deviceList = Object.values(devices);
            const totalDevices = deviceList.length;
            const healthyDevices = deviceList.filter(d => d.health.status === 'healthy').length;
            
            document.getElementById('total-devices').textContent = totalDevices;
            document.getElementById('healthy-devices').textContent = healthyDevices;
            
            const overallStatus = health ? health.overall_status : 'unknown';
            const statusElement = document.getElementById('overall-status');
            statusElement.textContent = overallStatus;
            statusElement.className = `metric-value status-${overallStatus}`;
        }
        
        function updateThroughput(throughput) {
            document.getElementById('requests-per-sec').textContent = 
                throughput.requests_per_second ? throughput.requests_per_second.toFixed(1) : '0';
            document.getElementById('tokens-per-sec').textContent = 
                throughput.tokens_per_second ? throughput.tokens_per_second.toFixed(1) : '0';
            document.getElementById('avg-latency').textContent = 
                throughput.avg_latency_ms ? throughput.avg_latency_ms.toFixed(0) : '0';
        }
        
        function updateStats(stats) {
            document.getElementById('total-requests').textContent = stats.requests_count || '0';
            document.getElementById('success-rate').textContent = 
                stats.success_rate ? (stats.success_rate * 100).toFixed(1) + '%' : '0%';
            document.getElementById('tokens-generated').textContent = 
                stats.total_tokens_generated || '0';
        }
        
        function updateDevices(devices) {
            const container = document.getElementById('devices-container');
            container.innerHTML = '';
            
            Object.values(devices).forEach(device => {
                const deviceCard = createDeviceCard(device);
                container.appendChild(deviceCard);
            });
        }
        
        function createDeviceCard(device) {
            const card = document.createElement('div');
            card.className = `device-card ${device.health.status}`;
            
            const cpuPercent = device.metrics.cpu_percent || 0;
            const memoryPercent = device.metrics.memory_percent || 0;
            
            card.innerHTML = `
                <div class="device-header">${device.device_id}</div>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="metric-value status-${device.health.status}">${device.health.status}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CPU</span>
                    <span class="metric-value">${cpuPercent.toFixed(1)}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill ${cpuPercent > 80 ? 'critical' : cpuPercent > 60 ? 'warning' : ''}" 
                         style="width: ${cpuPercent}%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory</span>
                    <span class="metric-value">${memoryPercent.toFixed(1)}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill ${memoryPercent > 80 ? 'critical' : memoryPercent > 60 ? 'warning' : ''}" 
                         style="width: ${memoryPercent}%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Model Loaded</span>
                    <span class="metric-value">${device.metrics.model_loaded ? 'Yes' : 'No'}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Layers</span>
                    <span class="metric-value">${device.metrics.assigned_layers.length}</span>
                </div>
            `;
            
            return card;
        }
        
        // Start the dashboard
        connectWebSocket();
        
        // Refresh data every 5 seconds as fallback
        setInterval(fetchData, 5000);
    </script>
</body>
</html>
        """
    
    async def start(self):
        """Start the monitoring dashboard."""
        # Start broadcasting task
        self._broadcast_task = asyncio.create_task(self._broadcast_updates())
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        logger.info(f"Starting monitoring dashboard at http://{self.host}:{self.port}")
        await server.serve()
    
    async def stop(self):
        """Stop the monitoring dashboard."""
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        
        # Close all WebSocket connections
        for ws in self.websocket_connections:
            await ws.close()
        self.websocket_connections.clear()
        
        logger.info("Stopped monitoring dashboard")
    
    async def _broadcast_updates(self):
        """Broadcast updates to connected WebSocket clients."""
        while True:
            try:
                if self.websocket_connections:
                    # Gather all monitoring data
                    data = {
                        'timestamp': time.time(),
                        'metrics': self.metrics_collector.export_metrics(),
                        'health': self.health_monitor.get_health_summary(),
                        'devices': await self._get_devices_data(),
                        'throughput': self.metrics_collector.get_throughput_metrics(),
                        'stats': self.metrics_collector.get_inference_stats(300)  # Last 5 minutes
                    }
                    
                    # Broadcast to all connected clients
                    message = json.dumps(data)
                    disconnected = []
                    
                    for ws in self.websocket_connections:
                        try:
                            await ws.send_text(message)
                        except Exception as e:
                            logger.warning(f"Failed to send WebSocket update: {e}")
                            disconnected.append(ws)
                    
                    # Remove disconnected clients
                    for ws in disconnected:
                        if ws in self.websocket_connections:
                            self.websocket_connections.remove(ws)
                
                await asyncio.sleep(2.0)  # Update every 2 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _get_devices_data(self) -> Dict[str, Any]:
        """Get combined device data."""
        device_metrics = self.metrics_collector.get_device_metrics()
        device_health = self.health_monitor.get_device_health()
        
        devices = {}
        all_device_ids = set(device_metrics.keys()) | set(device_health.keys())
        
        for device_id in all_device_ids:
            metrics = device_metrics.get(device_id)
            health = device_health.get(device_id)
            
            devices[device_id] = {
                'device_id': device_id,
                'metrics': {
                    'cpu_percent': metrics.cpu_percent if metrics else 0,
                    'memory_percent': metrics.memory_percent if metrics else 0,
                    'memory_used_gb': metrics.memory_used_gb if metrics else 0,
                    'memory_total_gb': metrics.memory_total_gb if metrics else 0,
                    'gpu_utilization': metrics.gpu_utilization if metrics else None,
                    'model_loaded': metrics.model_loaded if metrics else False,
                    'assigned_layers': metrics.assigned_layers if metrics else [],
                    'timestamp': metrics.timestamp if metrics else 0
                },
                'health': {
                    'status': health.status.value if health else 'unknown',
                    'is_responsive': health.is_responsive if health else False,
                    'last_seen': health.last_seen if health else 0,
                    'checks_count': len(health.checks) if health else 0
                }
            }
        
        return devices


async def start_monitoring_dashboard(metrics_collector: MetricsCollector,
                                   health_monitor: HealthMonitor,
                                   host: str = "0.0.0.0",
                                   port: int = 8080):
    """Start the monitoring dashboard."""
    dashboard = MonitoringDashboard(metrics_collector, health_monitor, host, port)
    await dashboard.start()