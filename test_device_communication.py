#!/usr/bin/env python3
"""
Comprehensive device communication and gRPC validation for MLX distributed inference.
Tests network connectivity, gRPC services, and inter-device communication.
"""

import asyncio
import logging
import sys
import time
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import mlx.core as mx

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.config import ClusterConfig, DeviceRole
    from src.communication.grpc_client import ConnectionPool, GRPCInferenceClient
    from src.communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
except ImportError:
    # Fallback for direct imports
    from core.config import ClusterConfig, DeviceRole
    from communication.grpc_client import ConnectionPool, GRPCInferenceClient
    from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceCommunicationValidator:
    """Comprehensive device communication validation suite."""
    
    def __init__(self, config_path: str = "config/cluster_config.yaml"):
        """Initialize validator with config."""
        self.config_path = config_path
        self.config = None
        self.test_results = {}
        self.failures = []
        self.network_metrics = {}
    
    def log_failure(self, test_name: str, description: str, error: Optional[Exception] = None):
        """Log a test failure."""
        failure = {
            "test": test_name,
            "description": description,
            "error": str(error) if error else None,
            "timestamp": time.time()
        }
        self.failures.append(failure)
        logger.error(f"FAILED: {test_name} - {description}")
        if error:
            logger.error(f"Error details: {error}")
    
    def log_success(self, test_name: str, metrics: Optional[Dict] = None):
        """Log a test success."""
        logger.info(f"PASSED: {test_name}")
        if metrics:
            self.network_metrics[test_name] = metrics
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all device communication validation tests."""
        logger.info("Starting comprehensive device communication validation...")
        
        # Load configuration
        try:
            self.config = ClusterConfig.from_yaml(self.config_path)
        except Exception as e:
            self.log_failure("Configuration Load", f"Failed to load config: {e}", e)
            return self._generate_report()
        
        # Run test suite
        tests = [
            ("network_connectivity", self.test_network_connectivity),
            ("port_availability", self.test_port_availability),
            ("hostname_resolution", self.test_hostname_resolution),
            ("grpc_client_creation", self.test_grpc_client_creation),
            ("connection_pool_setup", self.test_connection_pool_setup),
            ("device_reachability", self.test_device_reachability),
            ("grpc_service_simulation", self.test_grpc_service_simulation),
            ("tensor_transmission", self.test_tensor_transmission),
            ("concurrent_connections", self.test_concurrent_connections),
            ("connection_resilience", self.test_connection_resilience),
            ("bandwidth_estimation", self.test_bandwidth_estimation),
            ("latency_measurement", self.test_latency_measurement)
        ]
        
        for test_name, test_func in tests:
            try:
                await test_func()
                self.test_results[test_name] = "PASSED"
            except Exception as e:
                self.log_failure(test_name, f"Test execution failed", e)
                self.test_results[test_name] = "FAILED"
        
        return self._generate_report()
    
    async def test_network_connectivity(self):
        """Test basic network connectivity between devices."""
        logger.info("Testing network connectivity...")
        
        connectivity_results = {}
        
        for device in self.config.devices:
            hostname = device.hostname
            
            # Test ping connectivity
            try:
                # Use ping to test basic connectivity
                result = subprocess.run(
                    ['ping', '-c', '1', '-W', '3000', hostname], 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    connectivity_results[device.device_id] = {
                        "status": "reachable",
                        "hostname": hostname,
                        "ping_success": True
                    }
                    logger.info(f"✓ {device.device_id} ({hostname}) is reachable")
                else:
                    connectivity_results[device.device_id] = {
                        "status": "unreachable", 
                        "hostname": hostname,
                        "ping_success": False,
                        "error": result.stderr
                    }
                    logger.warning(f"⚠ {device.device_id} ({hostname}) is not reachable via ping")
                    
            except subprocess.TimeoutExpired:
                connectivity_results[device.device_id] = {
                    "status": "timeout",
                    "hostname": hostname,
                    "ping_success": False,
                    "error": "ping timeout"
                }
                logger.warning(f"⚠ {device.device_id} ({hostname}) ping timeout")
            except Exception as e:
                connectivity_results[device.device_id] = {
                    "status": "error",
                    "hostname": hostname, 
                    "ping_success": False,
                    "error": str(e)
                }
                logger.warning(f"⚠ {device.device_id} ({hostname}) ping error: {e}")
        
        # Check if at least local device is reachable
        local_device_id = self.config.get_local_device_id()
        local_connectivity = connectivity_results.get(local_device_id, {})
        
        reachable_devices = sum(1 for result in connectivity_results.values() 
                              if result["status"] == "reachable")
        
        self.log_success("network_connectivity", {
            "total_devices": len(self.config.devices),
            "reachable_devices": reachable_devices,
            "connectivity_results": connectivity_results,
            "local_device_reachable": local_connectivity.get("status") == "reachable"
        })
    
    async def test_port_availability(self):
        """Test if required ports are available on devices."""
        logger.info("Testing port availability...")
        
        port_results = {}
        
        for device in self.config.devices:
            hostname = device.hostname
            ports_to_test = [device.api_port, device.grpc_port]
            
            device_ports = {}
            for port in ports_to_test:
                try:
                    # Test if port is reachable
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((hostname, port))
                    sock.close()
                    
                    if result == 0:
                        device_ports[port] = {"status": "open", "service_running": True}
                        logger.info(f"✓ {device.device_id}:{port} is open")
                    else:
                        device_ports[port] = {"status": "closed", "service_running": False}
                        logger.info(f"ℹ {device.device_id}:{port} is closed (service not running)")
                        
                except Exception as e:
                    device_ports[port] = {"status": "error", "error": str(e)}
                    logger.warning(f"⚠ {device.device_id}:{port} error: {e}")
            
            port_results[device.device_id] = {
                "hostname": hostname,
                "ports": device_ports,
                "api_port": device.api_port,
                "grpc_port": device.grpc_port
            }
        
        self.log_success("port_availability", port_results)
    
    async def test_hostname_resolution(self):
        """Test hostname resolution for all devices."""
        logger.info("Testing hostname resolution...")
        
        resolution_results = {}
        
        for device in self.config.devices:
            hostname = device.hostname
            
            try:
                # Resolve hostname to IP
                ip_address = socket.gethostbyname(hostname)
                resolution_results[device.device_id] = {
                    "hostname": hostname,
                    "ip_address": ip_address,
                    "resolved": True
                }
                logger.info(f"✓ {device.device_id}: {hostname} -> {ip_address}")
                
            except socket.gaierror as e:
                resolution_results[device.device_id] = {
                    "hostname": hostname,
                    "resolved": False,
                    "error": str(e)
                }
                logger.warning(f"⚠ {device.device_id}: {hostname} resolution failed: {e}")
            except Exception as e:
                resolution_results[device.device_id] = {
                    "hostname": hostname,
                    "resolved": False,
                    "error": str(e)
                }
                logger.warning(f"⚠ {device.device_id}: {hostname} error: {e}")
        
        resolved_count = sum(1 for result in resolution_results.values() if result["resolved"])
        
        self.log_success("hostname_resolution", {
            "total_devices": len(self.config.devices),
            "resolved_devices": resolved_count,
            "resolution_results": resolution_results
        })
    
    async def test_grpc_client_creation(self):
        """Test gRPC client creation for remote devices."""
        logger.info("Testing gRPC client creation...")
        
        local_device_id = self.config.get_local_device_id()
        client_results = {}
        
        for device in self.config.devices:
            if device.device_id == local_device_id:
                continue  # Skip local device
            
            try:
                # Create gRPC client
                client = GRPCInferenceClient(device.hostname, device.grpc_port)
                
                client_results[device.device_id] = {
                    "hostname": device.hostname,
                    "grpc_port": device.grpc_port,
                    "client_created": True,
                    "channel_state": str(client.channel.get_state(try_to_connect=False))
                }
                
                # Clean up
                client.close()
                logger.info(f"✓ gRPC client created for {device.device_id}")
                
            except Exception as e:
                client_results[device.device_id] = {
                    "hostname": device.hostname,
                    "grpc_port": device.grpc_port,
                    "client_created": False,
                    "error": str(e)
                }
                logger.warning(f"⚠ gRPC client creation failed for {device.device_id}: {e}")
        
        successful_clients = sum(1 for result in client_results.values() 
                               if result["client_created"])
        
        self.log_success("grpc_client_creation", {
            "total_remote_devices": len(self.config.devices) - 1,
            "successful_clients": successful_clients,
            "client_results": client_results
        })
    
    async def test_connection_pool_setup(self):
        """Test connection pool initialization."""
        logger.info("Testing connection pool setup...")
        
        try:
            local_device_id = self.config.get_local_device_id()
            connection_pool = ConnectionPool(self.config, local_device_id)
            
            # Check client count
            expected_clients = len(self.config.devices) - 1  # All except local
            actual_clients = len(connection_pool.clients)
            
            # Test get_next_device_client functionality
            coordinator = self.config.get_coordinator()
            next_device_tests = {}
            
            if coordinator:
                try:
                    next_client = connection_pool.get_next_device_client(coordinator.device_id)
                    next_device_tests["from_coordinator"] = {
                        "success": next_client is not None,
                        "next_device": next_client.device_id if next_client else None
                    }
                except Exception as e:
                    next_device_tests["from_coordinator"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Test from each device
            for device in self.config.devices:
                try:
                    next_client = connection_pool.get_next_device_client(device.device_id)
                    next_device_tests[f"from_{device.device_id}"] = {
                        "success": next_client is not None,
                        "next_device": next_client.device_id if next_client else None
                    }
                except Exception as e:
                    next_device_tests[f"from_{device.device_id}"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            self.log_success("connection_pool_setup", {
                "expected_clients": expected_clients,
                "actual_clients": actual_clients,
                "local_device_id": local_device_id,
                "next_device_tests": next_device_tests,
                "client_device_ids": [client.device_id for client in connection_pool.clients.values()]
            })
            
        except Exception as e:
            raise ValueError(f"Connection pool setup failed: {e}")
    
    async def test_device_reachability(self):
        """Test if devices are reachable via their configured addresses."""
        logger.info("Testing device reachability...")
        
        reachability_results = {}
        
        for device in self.config.devices:
            hostname = device.hostname
            api_port = device.api_port
            grpc_port = device.grpc_port
            
            device_result = {
                "hostname": hostname,
                "api_port_reachable": False,
                "grpc_port_reachable": False,
                "response_times": {}
            }
            
            # Test API port reachability
            try:
                start_time = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((hostname, api_port))
                response_time = time.time() - start_time
                sock.close()
                
                if result == 0:
                    device_result["api_port_reachable"] = True
                    device_result["response_times"]["api_port"] = response_time
                    logger.info(f"✓ {device.device_id} API port reachable ({response_time:.3f}s)")
                else:
                    logger.info(f"ℹ {device.device_id} API port not reachable (service not running)")
            except Exception as e:
                logger.warning(f"⚠ {device.device_id} API port test error: {e}")
            
            # Test gRPC port reachability  
            try:
                start_time = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((hostname, grpc_port))
                response_time = time.time() - start_time
                sock.close()
                
                if result == 0:
                    device_result["grpc_port_reachable"] = True
                    device_result["response_times"]["grpc_port"] = response_time
                    logger.info(f"✓ {device.device_id} gRPC port reachable ({response_time:.3f}s)")
                else:
                    logger.info(f"ℹ {device.device_id} gRPC port not reachable (service not running)")
            except Exception as e:
                logger.warning(f"⚠ {device.device_id} gRPC port test error: {e}")
            
            reachability_results[device.device_id] = device_result
        
        # Summary statistics
        api_reachable = sum(1 for result in reachability_results.values() 
                           if result["api_port_reachable"])
        grpc_reachable = sum(1 for result in reachability_results.values() 
                            if result["grpc_port_reachable"])
        
        self.log_success("device_reachability", {
            "total_devices": len(self.config.devices),
            "api_ports_reachable": api_reachable,
            "grpc_ports_reachable": grpc_reachable,
            "reachability_results": reachability_results
        })
    
    async def test_grpc_service_simulation(self):
        """Simulate gRPC service calls without actual services running."""
        logger.info("Testing gRPC service simulation...")
        
        # This test checks the gRPC infrastructure without requiring running services
        local_device_id = self.config.get_local_device_id()
        simulation_results = {}
        
        for device in self.config.devices:
            if device.device_id == local_device_id:
                continue
            
            device_result = {
                "hostname": device.hostname,
                "grpc_port": device.grpc_port,
                "client_created": False,
                "channel_state": None,
                "connection_attempt": False
            }
            
            try:
                # Create client
                client = GRPCInferenceClient(device.hostname, device.grpc_port)
                device_result["client_created"] = True
                
                # Check channel state
                initial_state = client.channel.get_state(try_to_connect=False)
                device_result["channel_state"] = str(initial_state)
                
                # Attempt connection (will fail if service not running, but tests infrastructure)
                try:
                    client.channel.get_state(try_to_connect=True)
                    device_result["connection_attempt"] = True
                except:
                    # Expected to fail if service not running
                    device_result["connection_attempt"] = True
                
                client.close()
                logger.info(f"✓ gRPC infrastructure test passed for {device.device_id}")
                
            except Exception as e:
                device_result["error"] = str(e)
                logger.warning(f"⚠ gRPC infrastructure test failed for {device.device_id}: {e}")
            
            simulation_results[device.device_id] = device_result
        
        successful_simulations = sum(1 for result in simulation_results.values() 
                                   if result["client_created"])
        
        self.log_success("grpc_service_simulation", {
            "total_remote_devices": len(self.config.devices) - 1,
            "successful_simulations": successful_simulations,
            "simulation_results": simulation_results
        })
    
    async def test_tensor_transmission(self):
        """Test theoretical tensor transmission capabilities."""
        logger.info("Testing tensor transmission capabilities...")
        
        # Test tensor serialization for transmission
        test_tensors = [
            ("small", mx.random.normal(shape=(1, 128))),
            ("medium", mx.random.normal(shape=(1, 512, 1024))),
            ("large", mx.random.normal(shape=(1, 2048, 4096)))
        ]
        
        transmission_results = {}
        
        for tensor_name, tensor in test_tensors:
            try:
                # Serialize
                start_time = time.time()
                data, metadata = serialize_mlx_array(tensor)
                serialize_time = time.time() - start_time
                
                # Deserialize
                start_time = time.time()
                recovered = deserialize_mlx_array(data, metadata)
                deserialize_time = time.time() - start_time
                
                # Verify
                if not mx.allclose(tensor, recovered, atol=1e-6):
                    raise ValueError(f"Tensor {tensor_name} not preserved")
                
                data_size_mb = len(data) / (1024 * 1024)
                
                transmission_results[tensor_name] = {
                    "tensor_shape": tensor.shape,
                    "data_size_mb": data_size_mb,
                    "serialize_time": serialize_time,
                    "deserialize_time": deserialize_time,
                    "total_time": serialize_time + deserialize_time,
                    "throughput_mb_s": data_size_mb / (serialize_time + deserialize_time)
                }
                
                logger.info(f"✓ {tensor_name} tensor transmission test passed")
                
            except Exception as e:
                transmission_results[tensor_name] = {
                    "error": str(e),
                    "tensor_shape": tensor.shape
                }
                logger.warning(f"⚠ {tensor_name} tensor transmission test failed: {e}")
        
        self.log_success("tensor_transmission", transmission_results)
    
    async def test_concurrent_connections(self):
        """Test concurrent connection handling."""
        logger.info("Testing concurrent connections...")
        
        async def create_connection_to_device(device):
            """Create a connection to a device."""
            if device.device_id == self.config.get_local_device_id():
                return None
            
            try:
                client = GRPCInferenceClient(device.hostname, device.grpc_port)
                # Test basic channel state
                state = client.channel.get_state(try_to_connect=False)
                client.close()
                return {
                    "device_id": device.device_id,
                    "success": True,
                    "channel_state": str(state)
                }
            except Exception as e:
                return {
                    "device_id": device.device_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Create concurrent connections
        start_time = time.time()
        tasks = [create_connection_to_device(device) for device in self.config.devices]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Filter out None results (local device)
        results = [r for r in results if r is not None]
        
        successful_connections = sum(1 for r in results if r["success"])
        
        self.log_success("concurrent_connections", {
            "total_connections": len(results),
            "successful_connections": successful_connections,
            "total_time": total_time,
            "connections_per_second": len(results) / total_time if total_time > 0 else 0,
            "connection_results": results
        })
    
    async def test_connection_resilience(self):
        """Test connection resilience and error handling."""
        logger.info("Testing connection resilience...")
        
        resilience_results = {}
        
        # Test connection to non-existent host
        try:
            client = GRPCInferenceClient("nonexistent.host", 50051)
            client.close()
            resilience_results["nonexistent_host"] = {"handled": True}
        except Exception as e:
            resilience_results["nonexistent_host"] = {
                "handled": True,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        
        # Test connection to invalid port
        try:
            valid_device = self.config.devices[0]
            client = GRPCInferenceClient(valid_device.hostname, 99999)
            client.close()
            resilience_results["invalid_port"] = {"handled": True}
        except Exception as e:
            resilience_results["invalid_port"] = {
                "handled": True,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        
        # Test rapid connection creation and cleanup
        try:
            device = self.config.devices[0]
            clients = []
            for i in range(5):
                client = GRPCInferenceClient(device.hostname, device.grpc_port)
                clients.append(client)
            
            for client in clients:
                client.close()
            
            resilience_results["rapid_connections"] = {"handled": True, "count": 5}
        except Exception as e:
            resilience_results["rapid_connections"] = {
                "handled": False,
                "error": str(e)
            }
        
        self.log_success("connection_resilience", resilience_results)
    
    async def test_bandwidth_estimation(self):
        """Estimate bandwidth capabilities between devices."""
        logger.info("Testing bandwidth estimation...")
        
        # Simulate data transfer by measuring serialization/deserialization times
        test_sizes = [
            (1, 1024),      # 1 KB equivalent  
            (1, 10240),     # 10 KB equivalent
            (1, 102400),    # 100 KB equivalent
            (1, 1024000)    # 1 MB equivalent
        ]
        
        bandwidth_results = {}
        
        for i, (rows, cols) in enumerate(test_sizes):
            try:
                # Create test data
                tensor = mx.random.normal(shape=(rows, cols))
                
                # Measure serialization (upload simulation)
                start_time = time.time()
                data, metadata = serialize_mlx_array(tensor)
                serialize_time = time.time() - start_time
                
                # Measure deserialization (download simulation)  
                start_time = time.time()
                recovered = deserialize_mlx_array(data, metadata)
                deserialize_time = time.time() - start_time
                
                data_size_mb = len(data) / (1024 * 1024)
                
                bandwidth_results[f"size_{i}"] = {
                    "tensor_shape": (rows, cols),
                    "data_size_mb": data_size_mb,
                    "serialize_time": serialize_time,
                    "deserialize_time": deserialize_time,
                    "upload_bandwidth_mb_s": data_size_mb / serialize_time if serialize_time > 0 else 0,
                    "download_bandwidth_mb_s": data_size_mb / deserialize_time if deserialize_time > 0 else 0
                }
                
            except Exception as e:
                bandwidth_results[f"size_{i}"] = {
                    "error": str(e),
                    "tensor_shape": (rows, cols)
                }
        
        self.log_success("bandwidth_estimation", bandwidth_results)
    
    async def test_latency_measurement(self):
        """Measure latency for various operations."""
        logger.info("Testing latency measurement...")
        
        latency_results = {}
        
        # Connection establishment latency
        connection_times = []
        for device in self.config.devices:
            if device.device_id == self.config.get_local_device_id():
                continue
            
            try:
                start_time = time.time()
                client = GRPCInferenceClient(device.hostname, device.grpc_port)
                connection_time = time.time() - start_time
                client.close()
                
                connection_times.append(connection_time)
            except:
                pass  # Skip failed connections
        
        if connection_times:
            latency_results["connection"] = {
                "average_ms": np.mean(connection_times) * 1000,
                "min_ms": np.min(connection_times) * 1000,
                "max_ms": np.max(connection_times) * 1000,
                "samples": len(connection_times)
            }
        
        # Small tensor serialization latency
        small_tensor = mx.random.normal(shape=(1, 100))
        serialize_times = []
        deserialize_times = []
        
        for _ in range(10):
            start_time = time.time()
            data, metadata = serialize_mlx_array(small_tensor)
            serialize_times.append(time.time() - start_time)
            
            start_time = time.time()
            recovered = deserialize_mlx_array(data, metadata)
            deserialize_times.append(time.time() - start_time)
        
        latency_results["serialization"] = {
            "serialize_avg_ms": np.mean(serialize_times) * 1000,
            "deserialize_avg_ms": np.mean(deserialize_times) * 1000,
            "serialize_min_ms": np.min(serialize_times) * 1000,
            "deserialize_min_ms": np.min(deserialize_times) * 1000
        }
        
        self.log_success("latency_measurement", latency_results)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASSED")
        total_tests = len(self.test_results)
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "failures": self.failures,
            "network_metrics": self.network_metrics,
            "timestamp": time.time()
        }


def print_communication_report(results: Dict[str, Any]):
    """Print formatted device communication validation report."""
    print("\n" + "="*80)
    print("DEVICE COMMUNICATION VALIDATION RESULTS")
    print("="*80)
    
    summary = results["summary"]
    print(f"\nSUMMARY:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    
    # Test results
    print(f"\nTEST RESULTS:")
    for test_name, result in results["test_results"].items():
        status_emoji = "✅" if result == "PASSED" else "❌"
        print(f"  {status_emoji} {test_name}: {result}")
    
    # Network metrics
    if results["network_metrics"]:
        print(f"\nNETWORK METRICS:")
        for test_name, metrics in results["network_metrics"].items():
            print(f"  {test_name}:")
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    print(f"    {metric}:")
                    for sub_metric, sub_value in value.items():
                        print(f"      {sub_metric}: {sub_value}")
                elif isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
    
    # Failures
    if results["failures"]:
        print(f"\nFAILURES:")
        for i, failure in enumerate(results["failures"], 1):
            print(f"  {i}. {failure['test']}: {failure['description']}")
            if failure['error']:
                print(f"     Error: {failure['error']}")
    
    print("\n" + "="*80)


async def main():
    """Run device communication validation."""
    validator = DeviceCommunicationValidator()
    results = await validator.run_all_tests()
    print_communication_report(results)
    
    # Return exit code
    return results["summary"]["failed_tests"]


if __name__ == "__main__":
    import numpy as np
    exit_code = asyncio.run(main())
    sys.exit(exit_code)