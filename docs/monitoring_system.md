# MLX Distributed Inference Monitoring System

This document describes the comprehensive monitoring system implemented for the MLX distributed inference system.

## Overview

The monitoring system provides real-time visibility into the distributed inference pipeline, including:

- **Device Utilization**: CPU, memory, and GPU usage across all devices
- **Inference Metrics**: Throughput, latency, and token generation rates
- **Health Monitoring**: System health checks and failure detection
- **Network Connectivity**: Inter-device communication monitoring
- **Web Dashboard**: Real-time visualization and alerting

## Architecture

The monitoring system consists of several key components:

### Core Components

1. **MetricsCollector** (`src/monitoring/metrics.py`)
   - Collects device utilization metrics
   - Tracks inference performance
   - Records network connectivity metrics
   - Provides aggregated statistics

2. **HealthMonitor** (`src/monitoring/health.py`)
   - Performs system health checks
   - Monitors critical resources
   - Detects failure conditions
   - Provides health status reporting

3. **MonitoringDashboard** (`src/monitoring/dashboard.py`)
   - Web-based monitoring interface
   - Real-time data visualization
   - RESTful API for metrics access
   - WebSocket for live updates

### Integration Points

The monitoring system is integrated into the distributed inference pipeline at several key points:

- **Orchestrator**: Tracks request processing and device coordination
- **Worker Servers**: Monitors device status and layer processing
- **gRPC Clients**: Records network communication metrics

## Features

### Device Monitoring

- **CPU Utilization**: Real-time CPU usage percentage
- **Memory Usage**: Memory consumption and availability
- **GPU Metrics**: GPU utilization (where available)
- **Model Status**: Model loading and assignment status
- **Health Status**: Overall device health assessment

### Inference Monitoring

- **Request Tracking**: End-to-end request processing
- **Throughput Metrics**: Requests per second and tokens per second
- **Latency Analysis**: Per-device and total processing times
- **Error Tracking**: Request failures and error rates
- **Token Generation**: Token production statistics

### Network Monitoring

- **Connectivity Tests**: Inter-device communication health
- **Latency Measurement**: Network round-trip times
- **Success Rates**: Connection reliability metrics
- **Error Detection**: Network failure identification

### Health Checks

- **System Resources**: Memory, CPU, and disk space monitoring
- **Process Health**: Application process status
- **Service Availability**: Component availability checks
- **Distributed Health**: Multi-device health assessment

## Usage

### Basic Integration

```python
from src.monitoring import MetricsCollector, HealthMonitor

# Initialize monitoring components
metrics_collector = MetricsCollector(device_id="coordinator")
health_monitor = HealthMonitor(device_id="coordinator")

# Start monitoring
await metrics_collector.start()
await health_monitor.start()

# The components will automatically collect metrics
```

### Orchestrator Integration

```python
from src.coordinator.orchestrator import DistributedOrchestrator
from src.core.config import ClusterConfig

# Load configuration
config = ClusterConfig.from_yaml("config/cluster_config.yaml")

# Initialize orchestrator with monitoring enabled
orchestrator = DistributedOrchestrator(config, enable_monitoring=True)
await orchestrator.initialize()

# Monitoring happens automatically during inference
request = InferenceRequest(...)
response = await orchestrator.process_request(request)
```

### Worker Integration

```python
from src.worker.worker_server import WorkerServer

# Initialize worker with monitoring enabled
worker = WorkerServer("config/cluster_config.yaml", enable_monitoring=True)
await worker.start()

# Monitoring happens automatically
```

### Dashboard Usage

```python
from src.monitoring.dashboard import start_monitoring_dashboard

# Start monitoring dashboard
await start_monitoring_dashboard(
    metrics_collector,
    health_monitor,
    host="0.0.0.0",
    port=8080
)
```

### Standalone Dashboard

```bash
# Start standalone monitoring dashboard
python scripts/start_monitoring_dashboard.py --host 0.0.0.0 --port 8080
```

## API Endpoints

The monitoring dashboard provides RESTful API endpoints for accessing metrics:

- `GET /api/metrics` - Complete metrics export
- `GET /api/health` - Health status summary
- `GET /api/devices` - Device information and status
- `GET /api/throughput` - Throughput metrics
- `GET /api/stats` - Inference statistics
- `WebSocket /ws` - Real-time updates

### Example API Usage

```python
import aiohttp

async with aiohttp.ClientSession() as session:
    # Get current metrics
    async with session.get('http://localhost:8080/api/metrics') as resp:
        metrics = await resp.json()
    
    # Get health status
    async with session.get('http://localhost:8080/api/health') as resp:
        health = await resp.json()
    
    # Get throughput data
    async with session.get('http://localhost:8080/api/throughput') as resp:
        throughput = await resp.json()
```

## Configuration

### Metrics Collection

```python
# Customize collection interval
metrics_collector = MetricsCollector(
    device_id="device1",
    collection_interval=2.0  # Collect every 2 seconds
)

# Update device status
metrics_collector.update_device_status(
    device_id="device1",
    model_loaded=True,
    assigned_layers=[0, 1, 2],
    is_healthy=True
)
```

### Health Monitoring

```python
# Customize health check thresholds
health_monitor = HealthMonitor(device_id="device1")
health_monitor.memory_warning_threshold = 75.0  # 75%
health_monitor.memory_critical_threshold = 90.0  # 90%
health_monitor.cpu_warning_threshold = 70.0     # 70%

# Register custom health check
from monitoring.health import HealthCheck

def custom_check():
    # Custom health check logic
    return {
        'status': HealthStatus.HEALTHY,
        'message': 'Custom check passed',
        'details': {'custom_metric': 42}
    }

health_monitor.register_check(HealthCheck(
    name="custom_check",
    description="Custom application health check",
    check_function=custom_check,
    interval_seconds=30.0
))
```

### Dashboard Configuration

```python
# Custom dashboard configuration
dashboard = MonitoringDashboard(
    metrics_collector,
    health_monitor,
    host="0.0.0.0",
    port=8080
)

# Start dashboard
await dashboard.start()
```

## Monitoring Best Practices

### Performance Considerations

1. **Collection Interval**: Balance between data granularity and overhead
2. **Data Retention**: Limit historical data to manage memory usage
3. **Network Impact**: Monitor network overhead from health checks
4. **Resource Usage**: Ensure monitoring doesn't impact inference performance

### Alerting

While the monitoring system provides comprehensive data collection, consider implementing external alerting based on the API endpoints:

```python
# Example alerting logic
async def check_system_health():
    health_data = await get_health_data()
    
    if health_data['overall_status'] == 'critical':
        send_alert("System health critical!")
    
    throughput_data = await get_throughput_data()
    if throughput_data['requests_per_second'] < 1.0:
        send_alert("Low throughput detected!")
```

### Troubleshooting

1. **High Memory Usage**: Check metrics collection interval and data retention
2. **Network Issues**: Review gRPC client monitoring integration
3. **Health Check Failures**: Examine individual health check results
4. **Dashboard Performance**: Monitor WebSocket connection count

## Integration with External Systems

### Prometheus/Grafana

The monitoring system can be integrated with external monitoring systems:

```python
# Export metrics in Prometheus format
def export_prometheus_metrics():
    metrics = metrics_collector.export_metrics()
    # Convert to Prometheus format
    return prometheus_format(metrics)
```

### Log Aggregation

All monitoring components use Python logging for operational visibility:

```python
import logging

# Configure logging for monitoring
logging.getLogger('src.monitoring').setLevel(logging.INFO)
```

## Dependencies

The monitoring system requires the following additional dependencies:

- `psutil>=6.1.0` - System resource monitoring
- `fastapi>=0.115.0` - Web dashboard framework
- `uvicorn[standard]>=0.34.0` - ASGI server
- `jinja2>=3.1.0` - HTML templating
- `websockets>=12.0` - Real-time updates

These are automatically included in the project's `pyproject.toml`.

## Examples

See the following examples for complete usage demonstrations:

- `examples/monitoring_integration_example.py` - Comprehensive integration example
- `scripts/start_monitoring_dashboard.py` - Standalone dashboard startup

## Security Considerations

1. **Network Access**: Restrict dashboard access to trusted networks
2. **Authentication**: Consider adding authentication for production deployments
3. **Data Sensitivity**: Monitor what information is exposed via APIs
4. **Resource Limits**: Implement rate limiting for API endpoints

## Future Enhancements

Potential future improvements to the monitoring system:

1. **Distributed Dashboards**: Multiple dashboard instances for large clusters
2. **Advanced Alerting**: Built-in alerting rules and notification systems
3. **Historical Analysis**: Long-term data storage and trend analysis
4. **Performance Profiling**: Detailed performance bottleneck identification
5. **Auto-scaling Integration**: Metrics-based automatic scaling decisions