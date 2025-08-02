#!/usr/bin/env python3
"""
gRPC Tensor Service Implementation for Distributed MLX Inference
Real network communication between devices using gRPC
"""

import asyncio
import grpc
import logging
import time
from concurrent import futures
from typing import Dict, List, Optional
import threading

import mlx.core as mx
from mlx_lm import load

# Import generated gRPC stubs
import tensor_service_pb2
import tensor_service_pb2_grpc

# Import our communication module
from device_comm import TensorSerializer, TensorMessage

logger = logging.getLogger(__name__)

class TensorServiceImplementation(tensor_service_pb2_grpc.TensorServiceServicer):
    """gRPC service implementation for tensor operations"""
    
    def __init__(self, device_id: str, hostname: str, port: int, assigned_layers: List[int]):
        self.device_id = device_id
        self.hostname = hostname
        self.port = port
        self.assigned_layers = assigned_layers
        self.serializer = TensorSerializer()
        
        # Model components (loaded when needed)
        self.model = None
        self.tokenizer = None
        self.model_layers = {}
        
        # Tensor storage for inter-device communication
        self.tensor_storage: Dict[str, mx.array] = {}
        self.tensor_metadata: Dict[str, Dict] = {}
        
        # Performance tracking
        self.stats = {
            "tensors_sent": 0,
            "tensors_received": 0,
            "forward_passes": 0,
            "total_bytes_sent": 0,
            "total_bytes_received": 0
        }
        
        logger.info(f"Initialized TensorService for {device_id} on {hostname}:{port}")
        logger.info(f"Assigned layers: {assigned_layers}")
    
    def load_model_layers(self, model_name: str = "mlx-community/Qwen3-1.7B-8bit"):
        """Load only the assigned model layers"""
        try:
            logger.info(f"Loading model layers for {self.device_id}")
            
            # Load full model (in production, we'd only load assigned layers)
            self.model, self.tokenizer = load(model_name)
            logger.info(f"✅ Model loaded: {model_name}")
            
            # In a real implementation, we would extract only the assigned layers
            # For now, we'll simulate this by storing layer indices
            for layer_idx in self.assigned_layers:
                self.model_layers[layer_idx] = f"layer_{layer_idx}_placeholder"
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model layers: {e}")
            return False
    
    def SendTensor(self, request, context):
        """Receive tensor from another device"""
        try:
            tensor_data = request.tensor
            logger.info(f"Receiving tensor {tensor_data.tensor_id} from {tensor_data.device_source}")
            
            # Convert protobuf message to our TensorMessage format
            tensor_message = TensorMessage(
                tensor_id=tensor_data.tensor_id,
                shape=tuple(tensor_data.shape),
                dtype=tensor_data.dtype,
                data=tensor_data.data,
                compression=tensor_data.compression,
                device_source=tensor_data.device_source,
                device_target=tensor_data.device_target,
                timestamp=tensor_data.timestamp
            )
            
            # Deserialize tensor
            tensor = self.serializer.deserialize_tensor(tensor_message)
            
            # Store tensor
            self.tensor_storage[tensor_data.tensor_id] = tensor
            self.tensor_metadata[tensor_data.tensor_id] = {
                "source": tensor_data.device_source,
                "received_at": time.time(),
                "shape": tensor.shape,
                "dtype": str(tensor.dtype)
            }
            
            # Update stats
            self.stats["tensors_received"] += 1
            self.stats["total_bytes_received"] += len(tensor_data.data)
            
            logger.info(f"✅ Stored tensor {tensor_data.tensor_id} ({tensor.shape})")
            
            return tensor_service_pb2.SendTensorResponse(
                success=True,
                message=f"Tensor {tensor_data.tensor_id} received successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to receive tensor: {e}")
            return tensor_service_pb2.SendTensorResponse(
                success=False,
                message=f"Failed to receive tensor: {str(e)}"
            )
    
    def GetTensor(self, request, context):
        """Send tensor to requesting device"""
        try:
            tensor_id = request.tensor_id
            timeout = request.timeout
            
            logger.info(f"Request for tensor {tensor_id} (timeout: {timeout}s)")
            
            # Wait for tensor to be available
            start_time = time.time()
            while tensor_id not in self.tensor_storage:
                if time.time() - start_time > timeout:
                    return tensor_service_pb2.GetTensorResponse(
                        success=False,
                        message=f"Timeout waiting for tensor {tensor_id}"
                    )
                time.sleep(0.1)
            
            # Get tensor
            tensor = self.tensor_storage[tensor_id]
            
            # Serialize tensor
            tensor_message = self.serializer.serialize_tensor(tensor, tensor_id)
            tensor_message.device_source = self.device_id
            
            # Convert to protobuf format
            tensor_data = tensor_service_pb2.TensorData(
                tensor_id=tensor_message.tensor_id,
                shape=list(tensor_message.shape),
                dtype=tensor_message.dtype,
                data=tensor_message.data,
                compression=tensor_message.compression,
                device_source=tensor_message.device_source,
                device_target=tensor_message.device_target,
                timestamp=tensor_message.timestamp
            )
            
            # Update stats
            self.stats["tensors_sent"] += 1
            self.stats["total_bytes_sent"] += len(tensor_message.data)
            
            logger.info(f"✅ Sent tensor {tensor_id} ({tensor.shape})")
            
            return tensor_service_pb2.GetTensorResponse(
                success=True,
                message=f"Tensor {tensor_id} sent successfully",
                tensor=tensor_data
            )
            
        except Exception as e:
            logger.error(f"Failed to send tensor: {e}")
            return tensor_service_pb2.GetTensorResponse(
                success=False,
                message=f"Failed to send tensor: {str(e)}"
            )
    
    def HealthCheck(self, request, context):
        """Health check endpoint"""
        return tensor_service_pb2.HealthResponse(
            healthy=True,
            device_id=self.device_id,
            timestamp=time.time(),
            status_message="Service is healthy"
        )
    
    def ForwardPass(self, request, context):
        """Execute forward pass on assigned layers"""
        try:
            logger.info(f"Forward pass request: layers {request.layer_start}-{request.layer_end}")
            
            # Convert input tensor
            input_tensor_msg = TensorMessage(
                tensor_id=request.input_tensor.tensor_id,
                shape=tuple(request.input_tensor.shape),
                dtype=request.input_tensor.dtype,
                data=request.input_tensor.data,
                compression=request.input_tensor.compression
            )
            
            input_tensor = self.serializer.deserialize_tensor(input_tensor_msg)
            logger.info(f"  Input tensor: {input_tensor.shape}")
            
            # Execute forward pass (simplified simulation)
            output_tensor = input_tensor
            
            for layer_idx in range(request.layer_start, request.layer_end + 1):
                if layer_idx in self.assigned_layers:
                    # Simulate layer computation
                    output_tensor = mx.tanh(output_tensor)
                    logger.debug(f"  Processed layer {layer_idx}")
                else:
                    logger.warning(f"  Layer {layer_idx} not assigned to this device")
            
            logger.info(f"  Output tensor: {output_tensor.shape}")
            
            # Serialize output tensor
            output_msg = self.serializer.serialize_tensor(output_tensor, f"output_{request.pass_id}")
            output_msg.device_source = self.device_id
            
            # Convert to protobuf
            output_tensor_data = tensor_service_pb2.TensorData(
                tensor_id=output_msg.tensor_id,
                shape=list(output_msg.shape),
                dtype=output_msg.dtype,
                data=output_msg.data,
                compression=output_msg.compression,
                device_source=output_msg.device_source,
                timestamp=output_msg.timestamp
            )
            
            # Update stats
            self.stats["forward_passes"] += 1
            
            return tensor_service_pb2.ForwardPassResponse(
                success=True,
                message=f"Forward pass completed: layers {request.layer_start}-{request.layer_end}",
                output_tensor=output_tensor_data
            )
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            return tensor_service_pb2.ForwardPassResponse(
                success=False,
                message=f"Forward pass failed: {str(e)}"
            )
    
    def GetClusterStatus(self, request, context):
        """Return device status information"""
        device_info = tensor_service_pb2.DeviceInfo(
            device_id=self.device_id,
            hostname=self.hostname,
            port=self.port,
            role="coordinator" if self.device_id == "mini1" else "worker",
            layers=self.assigned_layers,
            status="online"
        )
        
        return tensor_service_pb2.ClusterStatusResponse(
            devices=[device_info],
            coordinator="mini1"
        )

class GRPCTensorClient:
    """gRPC client for communicating with other devices"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.connections: Dict[str, grpc.Channel] = {}
        self.stubs: Dict[str, tensor_service_pb2_grpc.TensorServiceStub] = {}
        self.serializer = TensorSerializer()
    
    async def connect_to_device(self, target_device: str, hostname: str, port: int) -> bool:
        """Connect to another device's gRPC service"""
        try:
            address = f"{hostname}:{port}"
            logger.info(f"Connecting to {target_device} at {address}")
            
            # Create gRPC channel
            channel = grpc.aio.insecure_channel(
                address,
                options=[
                    ('grpc.max_send_message_length', 512 * 1024 * 1024),  # 512MB
                    ('grpc.max_receive_message_length', 512 * 1024 * 1024),  # 512MB
                ]
            )
            
            # Create stub
            stub = tensor_service_pb2_grpc.TensorServiceStub(channel)
            
            # Test connection with health check
            request = tensor_service_pb2.HealthRequest(device_id=self.device_id)
            response = await stub.HealthCheck(request, timeout=5.0)
            
            if response.healthy:
                self.connections[target_device] = channel
                self.stubs[target_device] = stub
                logger.info(f"✅ Connected to {target_device}")
                return True
            else:
                logger.error(f"Health check failed for {target_device}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to {target_device}: {e}")
            return False
    
    async def send_tensor(self, tensor: mx.array, target_device: str, tensor_id: str) -> bool:
        """Send tensor to another device via gRPC"""
        try:
            if target_device not in self.stubs:
                logger.error(f"Not connected to {target_device}")
                return False
            
            # Serialize tensor
            tensor_msg = self.serializer.serialize_tensor(tensor, tensor_id)
            tensor_msg.device_source = self.device_id
            tensor_msg.device_target = target_device
            
            # Convert to protobuf
            tensor_data = tensor_service_pb2.TensorData(
                tensor_id=tensor_msg.tensor_id,
                shape=list(tensor_msg.shape),
                dtype=tensor_msg.dtype,
                data=tensor_msg.data,
                compression=tensor_msg.compression,
                device_source=tensor_msg.device_source,
                device_target=tensor_msg.device_target,
                timestamp=tensor_msg.timestamp
            )
            
            # Send via gRPC
            request = tensor_service_pb2.SendTensorRequest(tensor=tensor_data)
            stub = self.stubs[target_device]
            response = await stub.SendTensor(request, timeout=30.0)
            
            if response.success:
                logger.info(f"✅ Sent tensor {tensor_id} to {target_device}")
                return True
            else:
                logger.error(f"Failed to send tensor: {response.message}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending tensor to {target_device}: {e}")
            return False
    
    async def receive_tensor(self, tensor_id: str, source_device: str, timeout: float = 30.0) -> Optional[mx.array]:
        """Receive tensor from another device via gRPC"""
        try:
            if source_device not in self.stubs:
                logger.error(f"Not connected to {source_device}")
                return None
            
            # Request tensor
            request = tensor_service_pb2.GetTensorRequest(
                tensor_id=tensor_id,
                timeout=timeout
            )
            
            stub = self.stubs[source_device]
            response = await stub.GetTensor(request, timeout=timeout + 5.0)
            
            if response.success:
                # Deserialize tensor
                tensor_msg = TensorMessage(
                    tensor_id=response.tensor.tensor_id,
                    shape=tuple(response.tensor.shape),
                    dtype=response.tensor.dtype,
                    data=response.tensor.data,
                    compression=response.tensor.compression
                )
                
                tensor = self.serializer.deserialize_tensor(tensor_msg)
                logger.info(f"✅ Received tensor {tensor_id} from {source_device}")
                return tensor
            else:
                logger.error(f"Failed to receive tensor: {response.message}")
                return None
                
        except Exception as e:
            logger.error(f"Error receiving tensor from {source_device}: {e}")
            return None
    
    async def request_forward_pass(self, 
                                 input_tensor: mx.array, 
                                 target_device: str,
                                 layer_start: int, 
                                 layer_end: int,
                                 pass_id: str) -> Optional[mx.array]:
        """Request forward pass from another device"""
        try:
            if target_device not in self.stubs:
                logger.error(f"Not connected to {target_device}")
                return None
            
            # Serialize input tensor
            input_msg = self.serializer.serialize_tensor(input_tensor, f"input_{pass_id}")
            
            # Convert to protobuf
            input_tensor_data = tensor_service_pb2.TensorData(
                tensor_id=input_msg.tensor_id,
                shape=list(input_msg.shape),
                dtype=input_msg.dtype,
                data=input_msg.data,
                compression=input_msg.compression,
                device_source=self.device_id,
                timestamp=input_msg.timestamp
            )
            
            # Make forward pass request
            request = tensor_service_pb2.ForwardPassRequest(
                input_tensor=input_tensor_data,
                layer_start=layer_start,
                layer_end=layer_end,
                pass_id=pass_id
            )
            
            stub = self.stubs[target_device]
            response = await stub.ForwardPass(request, timeout=60.0)
            
            if response.success:
                # Deserialize output tensor
                output_msg = TensorMessage(
                    tensor_id=response.output_tensor.tensor_id,
                    shape=tuple(response.output_tensor.shape),
                    dtype=response.output_tensor.dtype,
                    data=response.output_tensor.data,
                    compression=response.output_tensor.compression
                )
                
                output_tensor = self.serializer.deserialize_tensor(output_msg)
                logger.info(f"✅ Forward pass completed on {target_device}")
                return output_tensor
            else:
                logger.error(f"Forward pass failed on {target_device}: {response.message}")
                return None
                
        except Exception as e:
            logger.error(f"Error requesting forward pass from {target_device}: {e}")
            return None

async def start_grpc_server(device_id: str, hostname: str, port: int, assigned_layers: List[int]):
    """Start gRPC server for this device"""
    try:
        # Create service implementation
        service = TensorServiceImplementation(device_id, hostname, port, assigned_layers)
        
        # Load model layers
        service.load_model_layers()
        
        # Create server
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 512 * 1024 * 1024),  # 512MB
                ('grpc.max_receive_message_length', 512 * 1024 * 1024),  # 512MB
            ]
        )
        
        # Add service to server
        tensor_service_pb2_grpc.add_TensorServiceServicer_to_server(service, server)
        
        # Listen on all interfaces
        listen_addr = f'0.0.0.0:{port}'
        server.add_insecure_port(listen_addr)
        
        logger.info(f"🚀 Starting gRPC TensorService for {device_id}")
        logger.info(f"   Listening on: {listen_addr}")
        logger.info(f"   Assigned layers: {assigned_layers}")
        
        # Start server
        await server.start()
        logger.info(f"✅ gRPC server started for {device_id}")
        
        # Keep server running
        await server.wait_for_termination()
        
    except Exception as e:
        logger.error(f"Failed to start gRPC server: {e}")
        raise

async def test_grpc_communication():
    """Test gRPC communication between devices"""
    logger.info("🧪 Testing gRPC communication...")
    
    # Start a test server in the background
    device_port = 50051
    server_task = asyncio.create_task(
        start_grpc_server("test_device", "localhost", device_port, [0, 1, 2])
    )
    
    # Give server time to start
    await asyncio.sleep(2)
    
    try:
        # Create client
        client = GRPCTensorClient("client_device")
        
        # Connect to server
        connected = await client.connect_to_device("test_device", "localhost", device_port)
        if not connected:
            logger.error("Failed to connect to test server")
            return False
        
        # Test tensor sending
        test_tensor = mx.random.normal((2, 128, 1536))
        success = await client.send_tensor(test_tensor, "test_device", "test_tensor_1")
        
        if success:
            logger.info("✅ gRPC tensor communication test passed")
            return True
        else:
            logger.error("❌ gRPC tensor communication test failed")
            return False
            
    except Exception as e:
        logger.error(f"gRPC communication test failed: {e}")
        return False
    finally:
        # Clean up
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test gRPC communication
    success = asyncio.run(test_grpc_communication())
    
    if success:
        print("🎉 gRPC tensor service implementation is working!")
    else:
        print("❌ gRPC tensor service test failed")