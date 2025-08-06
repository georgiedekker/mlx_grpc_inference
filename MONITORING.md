# Performance Monitoring Dashboard

## Overview

The MLX Distributed Inference system includes a comprehensive performance monitoring dashboard that provides real-time insights into system performance across all devices.

## Features

### Real-Time Metrics
- **Token Generation Rate**: Tracks tokens/second for both prompt evaluation and generation
- **Latency Monitoring**: Measures end-to-end latency and per-layer processing times
- **Memory Usage**: Monitors both system RAM and GPU memory usage
- **Network Bandwidth**: Tracks data transfer between devices
- **Cache Performance**: Monitors KV cache hit rates for efficient generation

### Performance Analysis
- Rolling window statistics (last 100 samples)
- Peak performance tracking
- Per-layer latency breakdown
- Session tracking and management

### Web Dashboard
- Real-time charts using Chart.js
- Auto-updating metrics every second
- Clean, modern UI with dark theme
- Responsive design for various screen sizes

## Architecture

```
┌──────────────────────────────────────┐
│         Performance Monitor          │
│  ┌─────────────┐  ┌────────────┐    │
│  │   Metrics   │  │   Layer    │    │
│  │  Collection │  │  Tracking  │    │
│  └─────────────┘  └────────────┘    │
│         ↓               ↓            │
│  ┌──────────────────────────────┐   │
│  │    Metrics Aggregation       │   │
│  └──────────────────────────────┘   │
│                ↓                     │
│  ┌──────────────────────────────┐   │
│  │    Dashboard Server          │   │
│  │  - FastAPI Backend           │   │
│  │  - HTML/JS Frontend          │   │
│  └──────────────────────────────┘   │
└──────────────────────────────────────┘
```

## Usage

### Starting with Monitoring

```bash
# Launch the complete system with monitoring
./launch_with_monitoring.sh
```

This will:
1. Start the performance monitoring dashboard on port 8888
2. Launch the distributed inference server on port 8000
3. Start remote workers on connected devices
4. Begin collecting performance metrics

### Standalone Dashboard

```bash
# Start just the dashboard
uv run python start_dashboard.py
```

### Accessing the Dashboard

Open your browser to: http://localhost:8888

## API Endpoints

The monitoring system provides several REST API endpoints:

### GET /api/metrics
Returns current performance metrics including:
- Current token generation rate
- Average metrics over last 20 samples
- Peak performance values
- Total tokens processed

### GET /api/layers
Returns per-layer performance statistics:
- Average processing time per layer
- Min/max/std deviation
- Percentiles (P50, P95, P99)

### GET /api/history
Returns complete metrics history for detailed analysis

## Integration with Server

The monitoring system is integrated into the main server code:

```python
from src.performance_monitor import get_monitor

# Get the singleton monitor instance
monitor = get_monitor()

# Record token generation
monitor.record_token_generation(
    session_id="uuid",
    num_tokens=50,
    time_taken=2.5,
    is_prompt=True
)

# Record layer processing
monitor.record_layer_processing(
    layer_idx=5,
    device_id="mini1",
    processing_time=0.015,
    tensor_size_bytes=1024*1024
)

# Track cache performance
monitor.record_cache_access(hit=True)

# Monitor network transfers
monitor.record_network_transfer(
    bytes_sent=1024*1024,
    bytes_received=512*1024
)
```

## Performance Metrics Explained

### Tokens/Second
- **Prompt Evaluation**: Speed of processing the initial prompt
- **Generation**: Speed of generating new tokens
- **Overall**: Combined throughput

### Latency
- **End-to-End**: Total time from request to response
- **Per-Layer**: Time spent in each transformer layer
- **Network**: Time spent in inter-device communication

### Memory Usage
- **System RAM**: Total system memory consumption
- **GPU Memory**: MLX metal memory usage
- **Per-Device**: Memory usage on each device

### Cache Performance
- **Hit Rate**: Percentage of cache hits vs misses
- **Cache Size**: Current KV cache memory usage
- **Efficiency**: Tokens generated per cache entry

## Optimizations Tracked

The monitoring system helps identify:
1. **Bottleneck Layers**: Which layers take the most time
2. **Network Overhead**: How much time is spent in communication
3. **Memory Pressure**: When devices are running low on memory
4. **Cache Efficiency**: Whether KV caching is working effectively
5. **Load Imbalance**: Whether work is evenly distributed

## Exporting Metrics

Export performance data for offline analysis:

```python
monitor = get_monitor()
monitor.export_metrics("/path/to/metrics.json")
```

The exported JSON includes:
- Complete metrics history
- Layer performance statistics
- Summary statistics

## Troubleshooting

### Dashboard Not Loading
1. Check if port 8888 is available: `lsof -i :8888`
2. Verify dependencies: `uv run python -c "import psutil, fastapi"`
3. Check logs: `tail -f dashboard.log`

### Metrics Not Updating
1. Ensure the server is running with monitoring enabled
2. Check that workers are connected
3. Verify network connectivity between devices

### High Memory Usage
The monitor keeps a rolling window of metrics. Adjust window size if needed:
```python
monitor = PerformanceMonitor(window_size=50)  # Smaller window
```

## Configuration

### Environment Variables
- `MONITOR_UPDATE_INTERVAL`: How often to collect metrics (default: 1.0 seconds)
- `MONITOR_WINDOW_SIZE`: Number of samples to keep (default: 100)
- `DASHBOARD_PORT`: Port for dashboard server (default: 8888)

### Customization
The dashboard HTML can be customized by modifying the `_generate_dashboard_html()` method in `DashboardServer`.

## Performance Impact

The monitoring system has minimal overhead:
- < 1% CPU usage for metric collection
- ~10MB RAM for rolling windows
- Negligible impact on inference speed

## Future Enhancements

Planned improvements:
- [ ] Grafana/Prometheus integration
- [ ] Historical data persistence
- [ ] Alert system for performance degradation
- [ ] Distributed tracing support
- [ ] Model-specific metrics
- [ ] A/B testing support