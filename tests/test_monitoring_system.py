#!/usr/bin/env python3
"""
Test script for the comprehensive monitoring system.

This script validates that all monitoring components are working correctly.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring import (
    MetricsCollector,
    HealthMonitor,
    MonitoringDashboard,
    HealthStatus
)


async def test_metrics_collector():
    """Test MetricsCollector functionality."""
    print("Testing MetricsCollector...")
    
    collector = MetricsCollector("test-device")
    
    try:
        # Start collector
        await collector.start()
        
        # Test request tracking
        metrics = collector.start_request("test-req-1", total_tokens=10)
        assert metrics.request_id == "test-req-1"
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Complete request
        collector.complete_request(
            "test-req-1",
            generated_tokens=5,
            device_times={"device1": 50.0},
            error=None
        )
        
        # Test network metric recording
        collector.record_network_metric("target-device", 25.0, True)
        
        # Test statistics
        stats = collector.get_inference_stats()
        assert stats['requests_count'] >= 1
        
        # Test throughput metrics
        throughput = collector.get_throughput_metrics()
        assert 'requests_per_second' in throughput
        
        # Test metrics export
        exported = collector.export_metrics()
        assert 'device_id' in exported
        assert exported['device_id'] == "test-device"
        
        print("✓ MetricsCollector tests passed")
        
    finally:
        await collector.stop()


async def test_health_monitor():
    """Test HealthMonitor functionality."""
    print("Testing HealthMonitor...")
    
    monitor = HealthMonitor("test-device")
    
    try:
        # Start monitor
        await monitor.start()
        
        # Wait for some health checks to run
        await asyncio.sleep(3)
        
        # Test health summary
        summary = monitor.get_health_summary()
        assert 'overall_status' in summary
        assert 'device_id' in summary
        assert summary['device_id'] == "test-device"
        
        # Test health check
        is_healthy = monitor.is_healthy()
        assert isinstance(is_healthy, bool)
        
        # Test device health retrieval
        device_health = monitor.get_device_health()
        assert "test-device" in device_health
        
        print("✓ HealthMonitor tests passed")
        
    finally:
        await monitor.stop()


async def test_dashboard_components():
    """Test dashboard components (without starting server)."""
    print("Testing MonitoringDashboard components...")
    
    collector = MetricsCollector("test-device")
    monitor = HealthMonitor("test-device")
    
    try:
        await collector.start()
        await monitor.start()
        
        # Create dashboard instance
        dashboard = MonitoringDashboard(collector, monitor, port=8081)
        
        # Test internal data methods
        devices_data = await dashboard._get_devices_data()
        assert isinstance(devices_data, dict)
        
        print("✓ MonitoringDashboard component tests passed")
        
    finally:
        await collector.stop()
        await monitor.stop()


async def test_integration():
    """Test integration between components."""
    print("Testing component integration...")
    
    collector = MetricsCollector("integration-test")
    monitor = HealthMonitor("integration-test")
    
    try:
        await collector.start()
        await monitor.start()
        
        # Update device status
        collector.update_device_status(
            "integration-test",
            model_loaded=True,
            assigned_layers=[0, 1, 2],
            is_healthy=True
        )
        
        # Simulate some activity
        for i in range(3):
            req_id = f"integration-req-{i}"
            collector.start_request(req_id, total_tokens=20)
            await asyncio.sleep(0.05)
            collector.complete_request(req_id, 10, {"device": 30.0})
            collector.record_network_metric("remote-device", 15.0, True)
        
        # Wait for health checks
        await asyncio.sleep(2)
        
        # Verify data
        stats = collector.get_inference_stats()
        assert stats['requests_count'] >= 3
        
        health = monitor.get_health_summary()
        assert health['overall_status'] in ['healthy', 'warning']
        
        metrics = collector.get_device_metrics()
        assert "integration-test" in metrics
        
        exported = collector.export_metrics()
        assert exported['inference_stats']['requests_count'] >= 3
        
        print("✓ Integration tests passed")
        
    finally:
        await collector.stop()
        await monitor.stop()


async def run_all_tests():
    """Run all monitoring system tests."""
    print("Starting monitoring system tests...\n")
    
    try:
        await test_metrics_collector()
        await test_health_monitor()
        await test_dashboard_components()
        await test_integration()
        
        print("\n✅ All monitoring system tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\nMonitoring system is ready for use!")
        sys.exit(0)
    else:
        print("\nMonitoring system tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()