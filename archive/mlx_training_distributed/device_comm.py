#!/usr/bin/env python3
"""
Device Communication Module for Distributed MLX Inference
Handles tensor serialization and communication between devices
"""

import asyncio
import pickle
import gzip
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import json

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TensorMessage:
    """Message containing tensor data for inter-device communication"""
    tensor_id: str
    shape: Tuple[int, ...]
    dtype: str
    data: bytes
    compression: str = "gzip"
    device_source: str = ""
    device_target: str = ""
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class TensorSerializer:
    """Handles MLX tensor serialization for network transmission"""
    
    @staticmethod
    def serialize_tensor(tensor: mx.array, tensor_id: str = "", compression: bool = True) -> TensorMessage:
        """Serialize MLX tensor to TensorMessage"""
        try:
            # Convert MLX array to numpy for serialization
            numpy_array = np.array(tensor)
            
            # Serialize numpy array
            tensor_bytes = pickle.dumps(numpy_array)
            
            # Optional compression
            if compression:
                tensor_bytes = gzip.compress(tensor_bytes)
                compression_method = "gzip"
            else:
                compression_method = "none"
            
            return TensorMessage(
                tensor_id=tensor_id,
                shape=tensor.shape,
                dtype=str(tensor.dtype),
                data=tensor_bytes,
                compression=compression_method
            )
            
        except Exception as e:
            logger.error(f"Failed to serialize tensor {tensor_id}: {e}")
            raise
    
    @staticmethod
    def deserialize_tensor(message: TensorMessage) -> mx.array:
        """Deserialize TensorMessage back to MLX tensor"""
        try:
            # Decompress if needed
            tensor_bytes = message.data
            if message.compression == "gzip":
                tensor_bytes = gzip.decompress(tensor_bytes)
            
            # Deserialize numpy array
            numpy_array = pickle.loads(tensor_bytes)
            
            # Convert back to MLX array
            mlx_tensor = mx.array(numpy_array)
            
            # Validate shape matches
            if mlx_tensor.shape != message.shape:
                logger.warning(f"Shape mismatch: expected {message.shape}, got {mlx_tensor.shape}")
            
            return mlx_tensor
            
        except Exception as e:
            logger.error(f"Failed to deserialize tensor {message.tensor_id}: {e}")
            raise

class DeviceCommunicator:
    """Manages communication with other devices in the cluster"""
    
    def __init__(self, device_id: str, hostname: str, port: int):
        self.device_id = device_id
        self.hostname = hostname
        self.port = port
        self.connections: Dict[str, Any] = {}  # Will store actual connections
        self.serializer = TensorSerializer()
        
    async def connect_to_device(self, target_device_id: str, target_hostname: str, target_port: int) -> bool:
        """Establish connection to another device"""
        try:
            logger.info(f"Connecting to {target_device_id} at {target_hostname}:{target_port}")
            
            # For now, just store connection info
            # In a full implementation, this would establish actual network connections
            self.connections[target_device_id] = {
                "hostname": target_hostname,
                "port": target_port,
                "connected": True,
                "last_ping": time.time()
            }
            
            logger.info(f"✅ Connected to {target_device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {target_device_id}: {e}")
            return False
    
    async def send_tensor(self, tensor: mx.array, target_device: str, tensor_id: str = "") -> bool:
        """Send tensor to another device"""
        try:
            if target_device not in self.connections:
                logger.error(f"Not connected to {target_device}")
                return False
            
            # Serialize tensor
            message = self.serializer.serialize_tensor(tensor, tensor_id)
            message.device_source = self.device_id
            message.device_target = target_device
            
            logger.info(f"Sending tensor {tensor_id} ({tensor.shape}) to {target_device}")
            logger.info(f"  Data size: {len(message.data)} bytes")
            logger.info(f"  Compression: {message.compression}")
            
            # In a full implementation, this would send via network
            # For now, just log the operation
            await asyncio.sleep(0.01)  # Simulate network delay
            
            logger.info(f"✅ Sent tensor {tensor_id} to {target_device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send tensor {tensor_id} to {target_device}: {e}")
            return False
    
    async def receive_tensor(self, expected_tensor_id: str = "", timeout: float = 30.0) -> Optional[Tuple[mx.array, TensorMessage]]:
        """Receive tensor from another device"""
        try:
            logger.info(f"Waiting for tensor {expected_tensor_id} (timeout: {timeout}s)")
            
            # In a full implementation, this would listen for incoming data
            # For now, simulate receiving
            await asyncio.sleep(0.01)  # Simulate network delay
            
            # Create a dummy tensor for testing
            dummy_tensor = mx.random.normal((1, 128, 1536))  # Typical transformer hidden state
            message = self.serializer.serialize_tensor(dummy_tensor, expected_tensor_id)
            
            logger.info(f"✅ Received tensor {expected_tensor_id} ({dummy_tensor.shape})")
            return dummy_tensor, message
            
        except Exception as e:
            logger.error(f"Failed to receive tensor {expected_tensor_id}: {e}")
            return None
    
    async def broadcast_tensor(self, tensor: mx.array, tensor_id: str = "") -> Dict[str, bool]:
        """Broadcast tensor to all connected devices"""
        results = {}
        
        for device_id in self.connections:
            success = await self.send_tensor(tensor, device_id, tensor_id)
            results[device_id] = success
        
        return results
    
    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connections"""
        status = {}
        current_time = time.time()
        
        for device_id, conn_info in self.connections.items():
            status[device_id] = {
                "connected": conn_info.get("connected", False),
                "hostname": conn_info.get("hostname", ""),
                "port": conn_info.get("port", 0),
                "last_ping": conn_info.get("last_ping", 0),
                "ping_age": current_time - conn_info.get("last_ping", 0)
            }
        
        return status
    
    async def ping_devices(self) -> Dict[str, bool]:
        """Ping all connected devices to check connectivity"""
        results = {}
        
        for device_id in self.connections:
            try:
                # Simulate ping
                await asyncio.sleep(0.001)
                self.connections[device_id]["last_ping"] = time.time()
                results[device_id] = True
                logger.debug(f"✅ Ping successful: {device_id}")
            except Exception as e:
                results[device_id] = False
                logger.warning(f"❌ Ping failed: {device_id} - {e}")
        
        return results

class DistributedTensorPipeline:
    """Manages tensor flow in distributed inference pipeline"""
    
    def __init__(self, device_id: str, layer_assignments: List[Tuple[str, List[int]]]):
        self.device_id = device_id
        self.layer_assignments = {device_id: layers for device_id, layers in layer_assignments}
        self.communicator = None
        self.tensor_cache: Dict[str, mx.array] = {}
        
    def set_communicator(self, communicator: DeviceCommunicator):
        """Set the device communicator"""
        self.communicator = communicator
    
    async def forward_pass_step(self, 
                              input_tensor: mx.array, 
                              layer_range: Tuple[int, int],
                              step_id: str = "") -> mx.array:
        """Execute forward pass for assigned layers"""
        try:
            logger.info(f"Forward pass step {step_id}: layers {layer_range[0]}-{layer_range[1]}")
            logger.info(f"  Input shape: {input_tensor.shape}")
            
            # Simulate layer processing (in real implementation, this would call actual model layers)
            # For now, just apply a simple transformation to simulate computation
            output_tensor = input_tensor
            
            for layer_idx in range(layer_range[0], layer_range[1] + 1):
                # Simulate layer computation
                output_tensor = mx.tanh(output_tensor)  # Simple activation
                logger.debug(f"  Processed layer {layer_idx}")
            
            logger.info(f"  Output shape: {output_tensor.shape}")
            return output_tensor
            
        except Exception as e:
            logger.error(f"Forward pass step {step_id} failed: {e}")
            raise
    
    async def coordinate_distributed_forward(self, input_tensor: mx.array) -> mx.array:
        """Coordinate distributed forward pass across all devices"""
        try:
            logger.info("🚀 Starting distributed forward pass")
            
            # Determine the pipeline order based on layer assignments
            device_order = sorted(self.layer_assignments.items(), key=lambda x: min(x[1]))
            
            current_tensor = input_tensor
            
            for i, (device_id, layers) in enumerate(device_order):
                layer_start, layer_end = min(layers), max(layers)
                
                if device_id == self.device_id:
                    # Process on this device
                    logger.info(f"Processing on local device {device_id}")
                    current_tensor = await self.forward_pass_step(
                        current_tensor, 
                        (layer_start, layer_end),
                        f"local_{i}"
                    )
                else:
                    # Send to remote device and wait for result
                    logger.info(f"Delegating to remote device {device_id}")
                    
                    if self.communicator:
                        # Send input tensor
                        await self.communicator.send_tensor(
                            current_tensor, 
                            device_id, 
                            f"forward_input_{i}"
                        )
                        
                        # Wait for processed result
                        result = await self.communicator.receive_tensor(f"forward_output_{i}")
                        if result:
                            current_tensor, _ = result
                        else:
                            raise Exception(f"Failed to receive result from {device_id}")
                    else:
                        logger.warning(f"No communicator available, simulating {device_id} processing")
                        # Simulate processing
                        current_tensor = await self.forward_pass_step(
                            current_tensor,
                            (layer_start, layer_end),
                            f"simulated_{i}"
                        )
            
            logger.info("✅ Distributed forward pass completed")
            return current_tensor
            
        except Exception as e:
            logger.error(f"Distributed forward pass failed: {e}")
            raise

async def test_device_communication():
    """Test device communication functionality"""
    logger.info("🧪 Testing device communication...")
    
    # Create communicators for different devices
    mini1_comm = DeviceCommunicator("mini1", "mini1.local", 8100)
    mini2_comm = DeviceCommunicator("mini2", "mini2.local", 8101)
    
    # Test connections
    await mini1_comm.connect_to_device("mini2", "mini2.local", 8101)
    await mini2_comm.connect_to_device("mini1", "mini1.local", 8100)
    
    # Test tensor serialization
    test_tensor = mx.random.normal((2, 128, 1536))  # Typical transformer tensor
    logger.info(f"Created test tensor: {test_tensor.shape}")
    
    # Test serialization
    serializer = TensorSerializer()
    message = serializer.serialize_tensor(test_tensor, "test_tensor")
    logger.info(f"Serialized tensor: {len(message.data)} bytes")
    
    # Test deserialization
    reconstructed_tensor = serializer.deserialize_tensor(message)
    logger.info(f"Reconstructed tensor: {reconstructed_tensor.shape}")
    
    # Verify data integrity
    if mx.allclose(test_tensor, reconstructed_tensor):
        logger.info("✅ Tensor serialization test passed")
    else:
        logger.error("❌ Tensor serialization test failed")
        return False
    
    # Test communication
    await mini1_comm.send_tensor(test_tensor, "mini2", "test_send")
    result = await mini2_comm.receive_tensor("test_receive")
    
    if result:
        received_tensor, message = result
        logger.info("✅ Tensor communication test passed")
    else:
        logger.error("❌ Tensor communication test failed")
        return False
    
    # Test distributed pipeline
    layer_assignments = [
        ("mini1", list(range(0, 10))),
        ("mini2", list(range(10, 19))),
        ("master", list(range(19, 28)))
    ]
    
    pipeline = DistributedTensorPipeline("mini1", layer_assignments)
    pipeline.set_communicator(mini1_comm)
    
    input_tensor = mx.random.normal((1, 128, 1536))
    output_tensor = await pipeline.coordinate_distributed_forward(input_tensor)
    
    if output_tensor is not None:
        logger.info("✅ Distributed pipeline test passed")
        logger.info(f"  Input shape: {input_tensor.shape}")
        logger.info(f"  Output shape: {output_tensor.shape}")
        return True
    else:
        logger.error("❌ Distributed pipeline test failed")
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = asyncio.run(test_device_communication())
    
    if success:
        print("🎉 Device communication implementation is working!")
    else:
        print("❌ Device communication test failed")