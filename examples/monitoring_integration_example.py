#!/usr/bin/env python3
"""
Example demonstrating comprehensive monitoring integration with MLX distributed inference.

This example shows how to:
1. Initialize monitoring components
2. Start monitoring dashboard
3. Integrate with orchestrator and workers
4. Query monitoring data
"""

import asyncio
import logging
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring import (
    MetricsCollector, 
    HealthMonitor, 
    MonitoringDashboard,
    DistributedHealthChecker
)
from core.config import ClusterConfig
from coordinator.orchestrator import DistributedOrchestrator, InferenceRequest


async def monitoring_integration_example():
    """Demonstrate monitoring integration."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("Starting monitoring integration example...")
    
    # Initialize monitoring components
    device_id = "coordinator-example"
    metrics_collector = MetricsCollector(device_id)
    health_monitor = HealthMonitor(device_id)
    
    try:
        # Start monitoring
        await metrics_collector.start()
        await health_monitor.start()
        logger.info("Started monitoring components")
        
        # Example: Record some mock metrics
        await simulate_inference_requests(metrics_collector, logger)
        
        # Example: Check health status
        await demonstrate_health_monitoring(health_monitor, logger)
        
        # Example: Get monitoring data
        await demonstrate_metrics_retrieval(metrics_collector, health_monitor, logger)
        
        # Start dashboard in background (for demonstration)
        logger.info("Starting monitoring dashboard at http://localhost:8081")
        dashboard = MonitoringDashboard(metrics_collector, health_monitor, port=8081)
        dashboard_task = asyncio.create_task(dashboard.start())
        
        # Let dashboard run for a short time
        await asyncio.sleep(5)
        
        # Cancel dashboard
        dashboard_task.cancel()
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass
        
        logger.info("Monitoring integration example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in monitoring example: {e}")
        raise
    finally:
        # Cleanup
        await metrics_collector.stop()
        await health_monitor.stop()
        logger.info("Stopped monitoring components")


async def simulate_inference_requests(metrics_collector: MetricsCollector, logger):
    """Simulate some inference requests for demonstration."""
    logger.info("Simulating inference requests...")
    
    # Simulate 5 requests
    for i in range(5):
        request_id = f"req_{i}"
        
        # Start request
        metrics = metrics_collector.start_request(request_id, total_tokens=50)
        
        # Simulate processing time
        await asyncio.sleep(0.1)  # 100ms processing
        
        # Simulate device processing times
        device_times = {
            "coordinator": 30.0,
            "worker1": 45.0,
            "worker2": 25.0
        }
        
        # Complete request
        metrics_collector.complete_request(
            request_id,
            generated_tokens=25,
            device_times=device_times,
            error=None if i < 4 else "Mock error for demonstration"
        )
        
        # Record network metrics
        metrics_collector.record_network_metric("worker1", 5.0, True)
        metrics_collector.record_network_metric("worker2", 7.5, True)
    
    logger.info("Simulated 5 inference requests")


async def demonstrate_health_monitoring(health_monitor: HealthMonitor, logger):
    """Demonstrate health monitoring capabilities."""
    logger.info("Demonstrating health monitoring...")
    
    # Wait a bit for health checks to run
    await asyncio.sleep(2)
    
    # Get health summary
    health_summary = health_monitor.get_health_summary()
    logger.info(f"Health status: {health_summary['overall_status']}")
    logger.info(f"Total health checks: {health_summary['total_checks']}")
    
    # Check if system is healthy
    is_healthy = health_monitor.is_healthy()
    logger.info(f"System is healthy: {is_healthy}")


async def demonstrate_metrics_retrieval(metrics_collector: MetricsCollector, 
                                      health_monitor: HealthMonitor, 
                                      logger):
    """Demonstrate retrieving various metrics."""
    logger.info("Demonstrating metrics retrieval...")
    
    # Get inference statistics
    stats = metrics_collector.get_inference_stats()
    logger.info(f"Inference stats: {stats['requests_count']} requests, "
               f"{stats['success_rate']:.1%} success rate")
    
    # Get throughput metrics
    throughput = metrics_collector.get_throughput_metrics(60)
    logger.info(f"Throughput: {throughput['requests_per_second']:.1f} req/s, "
               f"{throughput['tokens_per_second']:.1f} tokens/s")
    
    # Get network statistics
    network_stats = metrics_collector.get_network_stats()
    logger.info(f"Network: {network_stats['total_connections']} connections, "
               f"{network_stats['success_rate']:.1%} success rate")
    
    # Get device metrics
    device_metrics = metrics_collector.get_device_metrics()
    for device_id, metrics in device_metrics.items():
        logger.info(f"Device {device_id}: CPU {metrics.cpu_percent:.1f}%, "
                   f"Memory {metrics.memory_percent:.1f}%")
    
    # Export all metrics (useful for external monitoring systems)
    exported_metrics = metrics_collector.export_metrics()
    logger.info(f"Exported metrics contain {len(exported_metrics)} top-level keys")


async def demonstrate_orchestrator_integration():
    """Demonstrate how monitoring integrates with the orchestrator."""
    logger = logging.getLogger(__name__)
    
    # Note: This is a conceptual example showing how to integrate monitoring
    # In a real scenario, you would have a proper cluster configuration
    
    logger.info("Demonstrating orchestrator integration (conceptual)...")
    
    try:
        # Load cluster configuration (this would be a real config file)
        # config = ClusterConfig.from_yaml("config/cluster_config.yaml")
        
        # Initialize orchestrator with monitoring enabled
        # orchestrator = DistributedOrchestrator(config, enable_monitoring=True)
        # await orchestrator.initialize()
        
        # Get monitoring components from orchestrator
        # metrics_collector = orchestrator.get_metrics_collector()
        # health_monitor = orchestrator.get_health_monitor()
        
        # Process requests (monitoring happens automatically)
        # request = InferenceRequest(
        #     request_id="test_request",
        #     messages=[{"role": "user", "content": "Hello, world!"}]
        # )
        # response = await orchestrator.process_request(request)
        
        # Start monitoring dashboard
        # dashboard = MonitoringDashboard(metrics_collector, health_monitor)
        # await dashboard.start()
        
        logger.info("Orchestrator integration would be set up here")
        logger.info("Key integration points:")
        logger.info("1. Pass enable_monitoring=True to orchestrator")
        logger.info("2. Get monitoring components with get_metrics_collector()")
        logger.info("3. Start dashboard with collected metrics")
        logger.info("4. Monitoring happens automatically during inference")
        
    except Exception as e:
        logger.error(f"Error in orchestrator integration demo: {e}")


async def main():
    """Main entry point."""
    await monitoring_integration_example()
    await demonstrate_orchestrator_integration()


if __name__ == "__main__":
    asyncio.run(main())