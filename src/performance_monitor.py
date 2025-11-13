#!/usr/bin/env python3
"""
Performance monitoring dashboard for distributed MLX inference.
Tracks metrics, provides real-time visualization, and analyzes bottlenecks.
"""
import asyncio
import time
import psutil
import mlx.core as mx
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Any
from datetime import datetime
import logging
import json
from threading import Lock
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    tokens_per_second: float
    prompt_eval_tokens_per_second: float
    eval_tokens_per_second: float
    latency_ms: float
    memory_usage_mb: float
    gpu_memory_mb: float
    cpu_percent: float
    network_bandwidth_mbps: float
    cache_hit_rate: float
    active_sessions: int
    total_tokens_processed: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'tokens_per_second': self.tokens_per_second,
            'prompt_eval_tokens_per_second': self.prompt_eval_tokens_per_second,
            'eval_tokens_per_second': self.eval_tokens_per_second,
            'latency_ms': self.latency_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
            'cpu_percent': self.cpu_percent,
            'network_bandwidth_mbps': self.network_bandwidth_mbps,
            'cache_hit_rate': self.cache_hit_rate,
            'active_sessions': self.active_sessions,
            'total_tokens_processed': self.total_tokens_processed
        }


@dataclass 
class LayerMetrics:
    """Metrics for individual layer processing."""
    layer_idx: int
    device_id: str
    processing_time_ms: float
    tensor_size_mb: float
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class PerformanceMonitor:
    """Real-time performance monitoring for distributed inference."""
    
    def __init__(self, window_size: int = 100, update_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of samples to keep in rolling window
            update_interval: How often to update metrics (seconds)
        """
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Rolling windows for metrics
        self.metrics_history: Deque[PerformanceMetrics] = deque(maxlen=window_size)
        self.layer_metrics: Dict[int, Deque[LayerMetrics]] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Current session tracking
        self.active_sessions: Dict[str, Dict] = {}
        self.total_tokens_processed = 0
        self.network_bytes_sent = 0
        self.network_bytes_received = 0
        self._last_network_time = time.time()
        
        # Thread safety
        self._lock = Lock()
        
        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Layer timing
        self.layer_timings: Dict[int, List[float]] = defaultdict(list)
        
    def start(self):
        """Start background monitoring."""
        if not self._running:
            self._running = True
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                self._monitoring_task = loop.create_task(self._monitor_loop())
            except RuntimeError:
                # No event loop running - create monitoring task when one is available
                # This happens when monitor is created outside async context
                self._monitoring_task = None
            logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Log summary every 10 updates
                if len(self.metrics_history) % 10 == 0:
                    self._log_summary()
                    
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
        
        # GPU memory (MLX specific)
        try:
            gpu_memory_mb = mx.metal.get_memory_info()['current'] / (1024 * 1024)
        except:
            gpu_memory_mb = 0.0
        
        # Network bandwidth
        current_time = time.time()
        time_delta = current_time - self._last_network_time
        if time_delta > 0:
            bandwidth_mbps = ((self.network_bytes_sent + self.network_bytes_received) * 8) / (time_delta * 1e6)
            self.network_bytes_sent = 0
            self.network_bytes_received = 0
            self._last_network_time = current_time
        else:
            bandwidth_mbps = 0.0
        
        # Token processing metrics
        recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
        if recent_metrics:
            avg_tokens_per_second = np.mean([m.tokens_per_second for m in recent_metrics])
            avg_prompt_eval_tps = np.mean([m.prompt_eval_tokens_per_second for m in recent_metrics])
            avg_eval_tps = np.mean([m.eval_tokens_per_second for m in recent_metrics])
            avg_latency = np.mean([m.latency_ms for m in recent_metrics])
        else:
            avg_tokens_per_second = 0.0
            avg_prompt_eval_tps = 0.0
            avg_eval_tps = 0.0
            avg_latency = 0.0
        
        # Cache hit rate
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
        
        return PerformanceMetrics(
            timestamp=current_time,
            tokens_per_second=avg_tokens_per_second,
            prompt_eval_tokens_per_second=avg_prompt_eval_tps,
            eval_tokens_per_second=avg_eval_tps,
            latency_ms=avg_latency,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_mb=gpu_memory_mb,
            cpu_percent=cpu_percent,
            network_bandwidth_mbps=bandwidth_mbps,
            cache_hit_rate=cache_hit_rate,
            active_sessions=len(self.active_sessions),
            total_tokens_processed=self.total_tokens_processed
        )
    
    def record_token_generation(self, session_id: str, num_tokens: int, time_taken: float, is_prompt: bool = False):
        """Record token generation metrics."""
        with self._lock:
            self.total_tokens_processed += num_tokens
            tokens_per_second = num_tokens / time_taken if time_taken > 0 else 0
            
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    'start_time': time.time(),
                    'tokens_generated': 0,
                    'prompt_tokens': 0
                }
            
            session = self.active_sessions[session_id]
            session['tokens_generated'] += num_tokens
            
            if is_prompt:
                session['prompt_tokens'] += num_tokens
            
            # Create metrics entry
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                tokens_per_second=tokens_per_second,
                prompt_eval_tokens_per_second=tokens_per_second if is_prompt else 0,
                eval_tokens_per_second=tokens_per_second if not is_prompt else 0,
                latency_ms=time_taken * 1000,
                memory_usage_mb=0,  # Will be filled by monitor loop
                gpu_memory_mb=0,
                cpu_percent=0,
                network_bandwidth_mbps=0,
                cache_hit_rate=0,
                active_sessions=len(self.active_sessions),
                total_tokens_processed=self.total_tokens_processed
            )
            
            self.metrics_history.append(metrics)
    
    def record_layer_processing(self, layer_idx: int, device_id: str, processing_time: float, tensor_size_bytes: int):
        """Record layer processing metrics."""
        with self._lock:
            metrics = LayerMetrics(
                layer_idx=layer_idx,
                device_id=device_id,
                processing_time_ms=processing_time * 1000,
                tensor_size_mb=tensor_size_bytes / (1024 * 1024)
            )
            
            self.layer_metrics[layer_idx].append(metrics)
            self.layer_timings[layer_idx].append(processing_time * 1000)
    
    def record_cache_access(self, hit: bool):
        """Record cache hit or miss."""
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def record_network_transfer(self, bytes_sent: int = 0, bytes_received: int = 0):
        """Record network data transfer."""
        with self._lock:
            self.network_bytes_sent += bytes_sent
            self.network_bytes_received += bytes_received
    
    def end_session(self, session_id: str):
        """Mark session as completed."""
        with self._lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            recent = list(self.metrics_history)[-20:]  # Last 20 samples
            
            return {
                'current': self.metrics_history[-1].to_dict() if self.metrics_history else None,
                'averages': {
                    'tokens_per_second': np.mean([m.tokens_per_second for m in recent]),
                    'prompt_eval_tokens_per_second': np.mean([m.prompt_eval_tokens_per_second for m in recent]),
                    'eval_tokens_per_second': np.mean([m.eval_tokens_per_second for m in recent]),
                    'latency_ms': np.mean([m.latency_ms for m in recent]),
                    'memory_usage_mb': np.mean([m.memory_usage_mb for m in recent]),
                    'gpu_memory_mb': np.mean([m.gpu_memory_mb for m in recent]),
                    'cpu_percent': np.mean([m.cpu_percent for m in recent]),
                    'network_bandwidth_mbps': np.mean([m.network_bandwidth_mbps for m in recent]),
                    'cache_hit_rate': np.mean([m.cache_hit_rate for m in recent])
                },
                'peaks': {
                    'max_tokens_per_second': max([m.tokens_per_second for m in recent]),
                    'max_latency_ms': max([m.latency_ms for m in recent]),
                    'max_memory_usage_mb': max([m.memory_usage_mb for m in recent]),
                    'max_gpu_memory_mb': max([m.gpu_memory_mb for m in recent])
                },
                'totals': {
                    'total_tokens_processed': self.total_tokens_processed,
                    'active_sessions': len(self.active_sessions),
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses
                }
            }
    
    def get_layer_performance(self) -> Dict[int, Dict]:
        """Get per-layer performance statistics."""
        with self._lock:
            layer_stats = {}
            
            for layer_idx, timings in self.layer_timings.items():
                if timings:
                    layer_stats[layer_idx] = {
                        'avg_time_ms': np.mean(timings),
                        'min_time_ms': np.min(timings),
                        'max_time_ms': np.max(timings),
                        'std_time_ms': np.std(timings),
                        'p50_time_ms': np.percentile(timings, 50),
                        'p95_time_ms': np.percentile(timings, 95),
                        'p99_time_ms': np.percentile(timings, 99),
                        'samples': len(timings)
                    }
            
            return layer_stats
    
    def _log_summary(self):
        """Log performance summary."""
        summary = self.get_metrics_summary()
        if summary and 'current' in summary and summary['current']:
            current = summary['current']
            logger.info(
                f"Performance: {current['tokens_per_second']:.1f} tok/s | "
                f"Latency: {current['latency_ms']:.1f}ms | "
                f"GPU: {current['gpu_memory_mb']:.0f}MB | "
                f"Cache: {current['cache_hit_rate']*100:.1f}% | "
                f"Sessions: {current['active_sessions']}"
            )
    
    def export_metrics(self, filepath: str):
        """Export metrics history to JSON file."""
        with self._lock:
            data = {
                'metrics_history': [m.to_dict() for m in self.metrics_history],
                'layer_performance': self.get_layer_performance(),
                'summary': self.get_metrics_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")


class DashboardServer:
    """Web-based dashboard for real-time monitoring."""
    
    def __init__(self, monitor: PerformanceMonitor, port: int = 8080):
        """
        Initialize dashboard server.
        
        Args:
            monitor: PerformanceMonitor instance
            port: Port to serve dashboard on
        """
        self.monitor = monitor
        self.port = port
        self.app = None
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes for dashboard."""
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        
        self.app = FastAPI(title="MLX Distributed Inference Dashboard")
        
        @self.app.get("/")
        async def dashboard():
            """Serve dashboard HTML."""
            return HTMLResponse(self._generate_dashboard_html())
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics as JSON."""
            return JSONResponse(self.monitor.get_metrics_summary())
        
        @self.app.get("/api/layers")
        async def get_layer_metrics():
            """Get layer performance metrics."""
            return JSONResponse(self.monitor.get_layer_performance())
        
        @self.app.get("/api/history")
        async def get_history():
            """Get metrics history."""
            with self.monitor._lock:
                history = [m.to_dict() for m in self.monitor.metrics_history]
            return JSONResponse({"history": history})
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML with embedded JavaScript."""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>MLX Distributed Inference Dashboard</title>
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
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-healthy { background: #4ade80; }
        .status-warning { background: #fbbf24; }
        .status-error { background: #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 MLX Distributed Inference Dashboard</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">
                Real-time performance monitoring across devices
            </p>
        </div>
        
        <div class="metrics-grid">
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
                <div class="metric-value" id="cache-hit-rate">0</div>
                <div class="metric-label">Cache Hit Rate (%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="active-sessions">0</div>
                <div class="metric-label">Active Sessions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="total-tokens">0</div>
                <div class="metric-label">Total Tokens</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="performance-chart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="layer-chart"></canvas>
        </div>
    </div>
    
    <script>
        // Initialize charts
        const performanceCtx = document.getElementById('performance-chart').getContext('2d');
        const performanceChart = new Chart(performanceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Tokens/Second',
                    data: [],
                    borderColor: '#4ade80',
                    tension: 0.1
                }, {
                    label: 'Latency (ms)',
                    data: [],
                    borderColor: '#f87171',
                    tension: 0.1,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: { color: '#3a3a3a' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });
        
        // Update metrics every second
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
                    document.getElementById('cache-hit-rate').textContent = 
                        (data.current.cache_hit_rate * 100).toFixed(1);
                    document.getElementById('active-sessions').textContent = 
                        data.current.active_sessions;
                    document.getElementById('total-tokens').textContent = 
                        data.current.total_tokens_processed.toLocaleString();
                    
                    // Update chart
                    const time = new Date().toLocaleTimeString();
                    performanceChart.data.labels.push(time);
                    performanceChart.data.datasets[0].data.push(data.current.tokens_per_second);
                    performanceChart.data.datasets[1].data.push(data.current.latency_ms);
                    
                    // Keep only last 30 points
                    if (performanceChart.data.labels.length > 30) {
                        performanceChart.data.labels.shift();
                        performanceChart.data.datasets.forEach(d => d.data.shift());
                    }
                    
                    performanceChart.update();
                }
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
            }
        }
        
        // Start updating
        setInterval(updateMetrics, 1000);
        updateMetrics();
    </script>
</body>
</html>
        '''
    
    async def start(self):
        """Start dashboard server."""
        import uvicorn
        config = uvicorn.Config(app=self.app, host="0.0.0.0", port=self.port, log_level="error")
        server = uvicorn.Server(config)
        await server.serve()


# Singleton monitor instance
_monitor_instance: Optional[PerformanceMonitor] = None

def get_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
        # Don't auto-start - let the caller decide when to start
        # This avoids issues with event loops
    return _monitor_instance