"""
Distributed Communication Layer

This module provides abstraction for communication between devices
using gRPC as the communication backend.
"""

import mlx.core as mx
from typing import Any, Optional, List, Union, Dict, Tuple
from enum import Enum
import pickle
import numpy as np
import logging
import time
from abc import ABC, abstractmethod
import grpc
from concurrent import futures
import threading
import queue
import uuid
import os

# Import generated gRPC code
import distributed_comm_pb2
import distributed_comm_pb2_grpc

logger = logging.getLogger(__name__)


class CommunicationType(Enum):
    """Types of data that can be communicated."""
    TENSOR = "tensor"
    PICKLE = "pickle"
    NUMPY = "numpy"


class CommunicationBackend(Enum):
    """Available communication backends."""
    GRPC = "grpc"


class DistributedCommunicator(ABC):
    """Abstract base class for distributed communication."""
    
    @abstractmethod
    def init(self, rank: int, world_size: int) -> None:
        """Initialize the communication backend."""
        pass
    
    @abstractmethod
    def send(self, data: Any, dest: int, comm_type: CommunicationType = CommunicationType.PICKLE) -> None:
        """Send data to a specific destination."""
        pass
    
    @abstractmethod
    def receive(self, source: int, comm_type: CommunicationType = CommunicationType.PICKLE) -> Any:
        """Receive data from a specific source."""
        pass
    
    @abstractmethod
    def broadcast(self, data: Any, root: int, comm_type: CommunicationType = CommunicationType.PICKLE) -> Any:
        """Broadcast data from root to all processes."""
        pass
    
    @abstractmethod
    def allreduce(self, data: mx.array, op: str = "sum") -> mx.array:
        """Perform allreduce operation on data."""
        pass
    
    @abstractmethod
    def barrier(self) -> None:
        """Synchronization barrier."""
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """Cleanup communication backend."""
        pass


class GRPCCommServicer(distributed_comm_pb2_grpc.DistributedCommServicer):
    """gRPC service implementation for distributed communication."""
    
    def __init__(self, communicator: 'GRPCCommunicator'):
        self.communicator = communicator
        self.receive_queues: Dict[str, queue.Queue] = {}
        self.broadcast_data: Dict[str, Any] = {}
        self.allreduce_data: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        self.barrier_participants: Dict[str, set] = {}
        self.lock = threading.Lock()
    
    def Send(self, request, context):
        """Handle incoming send requests."""
        try:
            # Create queue key
            queue_key = f"{request.dest_rank}_{request.source_rank}_{request.tag}"
            
            # Store the data in the appropriate queue
            with self.lock:
                if queue_key not in self.receive_queues:
                    self.receive_queues[queue_key] = queue.Queue()
                self.receive_queues[queue_key].put(request.data)
            
            return distributed_comm_pb2.SendResponse(success=True, message="Data sent successfully")
        except Exception as e:
            logger.error(f"Send failed: {str(e)}")
            return distributed_comm_pb2.SendResponse(success=False, message=str(e))
    
    def Receive(self, request, context):
        """Handle receive requests (blocking)."""
        queue_key = f"{request.receiver_rank}_{request.source_rank}_{request.tag}"
        
        # Wait for data to arrive
        timeout = 60  # 60 second timeout
        start_time = time.time()
        
        while True:
            with self.lock:
                if queue_key in self.receive_queues and not self.receive_queues[queue_key].empty():
                    data = self.receive_queues[queue_key].get()
                    response = distributed_comm_pb2.ReceiveResponse(
                        data=data,
                        source_rank=request.source_rank,
                        tag=request.tag
                    )
                    yield response
                    break
            
            if time.time() - start_time > timeout:
                if context is not None:
                    context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Receive timeout")
                break
            
            time.sleep(0.001)  # Small sleep to avoid busy waiting
    
    def Broadcast(self, request, context):
        """Handle broadcast requests."""
        try:
            broadcast_id = f"broadcast_{request.root_rank}_{time.time()}"
            
            with self.lock:
                if request.sender_rank == request.root_rank:
                    # Root stores the data
                    self.broadcast_data[broadcast_id] = request.data
                else:
                    # Non-root waits for data
                    timeout = 10
                    start_time = time.time()
                    while broadcast_id not in self.broadcast_data:
                        if time.time() - start_time > timeout:
                            return distributed_comm_pb2.BroadcastResponse(
                                success=False,
                                data=None
                            )
                        time.sleep(0.001)
                    
                    data = self.broadcast_data[broadcast_id]
                    return distributed_comm_pb2.BroadcastResponse(
                        success=True,
                        data=data
                    )
            
            return distributed_comm_pb2.BroadcastResponse(
                success=True,
                data=request.data if request.sender_rank == request.root_rank else None
            )
        except Exception as e:
            logger.error(f"Broadcast failed: {str(e)}")
            return distributed_comm_pb2.BroadcastResponse(success=False, data=None)
    
    def AllReduce(self, request, context):
        """Handle allreduce requests."""
        try:
            allreduce_id = f"allreduce_{time.time()}"
            
            # Store this rank's contribution
            with self.lock:
                if allreduce_id not in self.allreduce_data:
                    self.allreduce_data[allreduce_id] = []
                
                # Deserialize tensor
                np_array = np.frombuffer(
                    request.tensor.data, 
                    dtype=request.tensor.dtype
                ).reshape(request.tensor.shape)
                
                self.allreduce_data[allreduce_id].append((request.rank, np_array))
                
                # Wait for all ranks
                expected_ranks = self.communicator.world_size
                timeout = 10
                start_time = time.time()
                
                while len(self.allreduce_data[allreduce_id]) < expected_ranks:
                    if time.time() - start_time > timeout:
                        return distributed_comm_pb2.AllReduceResponse(
                            success=False,
                            result=None
                        )
                    time.sleep(0.001)
                
                # Perform reduction
                arrays = [arr for _, arr in self.allreduce_data[allreduce_id]]
                
                if request.operation == "sum":
                    result = np.sum(arrays, axis=0)
                elif request.operation == "mean":
                    result = np.mean(arrays, axis=0)
                elif request.operation == "max":
                    result = np.maximum.reduce(arrays)
                elif request.operation == "min":
                    result = np.minimum.reduce(arrays)
                else:
                    raise ValueError(f"Unknown operation: {request.operation}")
                
                # Serialize result
                result_tensor = distributed_comm_pb2.TensorData(
                    data=result.tobytes(),
                    shape=list(result.shape),
                    dtype=str(result.dtype)
                )
                
                return distributed_comm_pb2.AllReduceResponse(
                    success=True,
                    result=result_tensor
                )
        except Exception as e:
            logger.error(f"AllReduce failed: {str(e)}")
            return distributed_comm_pb2.AllReduceResponse(success=False, result=None)
    
    def Barrier(self, request, context):
        """Handle barrier synchronization requests."""
        try:
            with self.lock:
                barrier_id = request.barrier_id
                if barrier_id not in self.barrier_participants:
                    self.barrier_participants[barrier_id] = set()
                
                self.barrier_participants[barrier_id].add(request.rank)
                
                # Wait for all ranks
                expected_ranks = self.communicator.world_size
                timeout = 30
                start_time = time.time()
                
                while len(self.barrier_participants[barrier_id]) < expected_ranks:
                    if time.time() - start_time > timeout:
                        return distributed_comm_pb2.BarrierResponse(
                            success=False,
                            participants=len(self.barrier_participants[barrier_id])
                        )
                    time.sleep(0.001)
                
                # Cleanup old barrier data
                if barrier_id in self.barrier_participants:
                    participants = len(self.barrier_participants[barrier_id])
                    del self.barrier_participants[barrier_id]
                else:
                    participants = expected_ranks
                
                return distributed_comm_pb2.BarrierResponse(
                    success=True,
                    participants=participants
                )
        except Exception as e:
            logger.error(f"Barrier failed: {str(e)}")
            return distributed_comm_pb2.BarrierResponse(success=False, participants=0)


class GRPCCommunicator(DistributedCommunicator):
    """gRPC-based communication implementation."""
    
    def __init__(self):
        self.rank = None
        self.world_size = None
        self._initialized = False
        self._server = None
        self._stubs = {}
        self._channels = {}
        self._servicer = None
        self._base_port = 50100  # Base port for communication service
        self._channel_options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.dns_resolver', 'native'),  # Use system DNS resolver instead of C-ares
        ]
    
    def init(self, rank: int, world_size: int, port: Optional[int] = None, device_hostnames: Optional[List[str]] = None) -> None:
        """Initialize gRPC communication."""
        self.rank = rank
        self.world_size = world_size
        self._initialized = True
        self.device_hostnames = device_hostnames or [f"localhost" for _ in range(world_size)]
        
        # Use provided port or calculate from base port
        comm_port = port if port is not None else self._base_port + rank
        
        # Start gRPC server
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self._servicer = GRPCCommServicer(self)
        distributed_comm_pb2_grpc.add_DistributedCommServicer_to_server(
            self._servicer, self._server
        )
        self._server.add_insecure_port(f'[::]:{comm_port}')
        self._server.start()
        
        # Create stubs for other processes using proper hostnames
        for i in range(world_size):
            if i != rank:
                target_port = self._base_port + i
                target_hostname = self.device_hostnames[i]
                channel = grpc.insecure_channel(
                    f'{target_hostname}:{target_port}', 
                    options=self._channel_options
                )
                self._channels[i] = channel
                self._stubs[i] = distributed_comm_pb2_grpc.DistributedCommStub(channel)
                logger.info(f"Connected to rank {i} at {target_hostname}:{target_port}")
        
        logger.info(f"gRPC communicator initialized: rank={self.rank}, world_size={self.world_size}, port={comm_port}")
    
    def _serialize_data(self, data: Any, comm_type: CommunicationType) -> distributed_comm_pb2.CommData:
        """Serialize data for gRPC transmission."""
        comm_data = distributed_comm_pb2.CommData(comm_type=comm_type.value)
        
        if comm_type == CommunicationType.TENSOR:
            if isinstance(data, mx.array):
                np_array = np.array(data)
                tensor_data = distributed_comm_pb2.TensorData(
                    data=np_array.tobytes(),
                    shape=list(np_array.shape),
                    dtype=str(np_array.dtype)
                )
                comm_data.tensor_data.CopyFrom(tensor_data)
            else:
                raise TypeError(f"Expected mx.array for TENSOR type, got {type(data)}")
        elif comm_type == CommunicationType.NUMPY:
            if isinstance(data, np.ndarray):
                comm_data.numpy_data = data.tobytes()
                comm_data.metadata['shape'] = str(list(data.shape))
                comm_data.metadata['dtype'] = str(data.dtype)
            else:
                raise TypeError(f"Expected np.ndarray for NUMPY type, got {type(data)}")
        else:  # PICKLE
            comm_data.pickle_data = pickle.dumps(data)
        
        return comm_data
    
    def _deserialize_data(self, comm_data: distributed_comm_pb2.CommData) -> Any:
        """Deserialize data from gRPC transmission."""
        comm_type = CommunicationType(comm_data.comm_type)
        
        if comm_type == CommunicationType.TENSOR:
            tensor_data = comm_data.tensor_data
            np_array = np.frombuffer(
                tensor_data.data, 
                dtype=tensor_data.dtype
            ).reshape(tensor_data.shape)
            return mx.array(np_array)
        elif comm_type == CommunicationType.NUMPY:
            shape = eval(comm_data.metadata['shape'])
            dtype = comm_data.metadata['dtype']
            return np.frombuffer(comm_data.numpy_data, dtype=dtype).reshape(shape)
        else:  # PICKLE
            return pickle.loads(comm_data.pickle_data)
    
    def send(self, data: Any, dest: int, comm_type: CommunicationType = CommunicationType.PICKLE) -> None:
        """Send data to a specific destination."""
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        if dest not in self._stubs:
            raise ValueError(f"No connection to rank {dest}")
        
        # Serialize data
        comm_data = self._serialize_data(data, comm_type)
        
        # Create request
        request = distributed_comm_pb2.SendRequest(
            source_rank=self.rank,
            dest_rank=dest,
            data=comm_data,
            tag=""
        )
        
        # Send via gRPC
        try:
            response = self._stubs[dest].Send(request)
            if not response.success:
                raise RuntimeError(f"Send failed: {response.message}")
            logger.debug(f"Sent data to rank {dest} (type: {comm_type})")
        except grpc.RpcError as e:
            logger.error(f"gRPC send error: {e}")
            raise
    
    def receive(self, source: int, comm_type: CommunicationType = CommunicationType.PICKLE) -> Any:
        """Receive data from a specific source."""
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        # Create request
        request = distributed_comm_pb2.ReceiveRequest(
            receiver_rank=self.rank,
            source_rank=source,
            tag="",
            comm_type=comm_type.value
        )
        
        # Receive via gRPC (blocking)
        try:
            # Get local servicer directly for same-process communication
            if hasattr(self, '_servicer'):
                for response in self._servicer.Receive(request, None):
                    data = self._deserialize_data(response.data)
                    logger.debug(f"Received data from rank {source} (type: {comm_type})")
                    return data
            else:
                raise RuntimeError("Servicer not initialized")
        except Exception as e:
            logger.error(f"Receive error: {e}")
            raise
    
    def broadcast(self, data: Any, root: int, comm_type: CommunicationType = CommunicationType.PICKLE) -> Any:
        """Broadcast data from root to all processes."""
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        if self.rank == root:
            # Root sends to all others
            for dest in range(self.world_size):
                if dest != root:
                    self.send(data, dest, comm_type)
            return data
        else:
            # Non-root receives from root
            return self.receive(root, comm_type)
    
    def allreduce(self, data: mx.array, op: str = "sum") -> mx.array:
        """Perform allreduce operation on MLX array."""
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        # Convert to numpy for processing
        np_data = np.array(data)
        
        # Create tensor data
        tensor_data = distributed_comm_pb2.TensorData(
            data=np_data.tobytes(),
            shape=list(np_data.shape),
            dtype=str(np_data.dtype)
        )
        
        # Create request
        request = distributed_comm_pb2.AllReduceRequest(
            rank=self.rank,
            tensor=tensor_data,
            operation=op
        )
        
        # Perform allreduce
        try:
            # For now, use a simple gather-reduce-scatter approach
            # In a real implementation, this would be optimized
            all_data = []
            
            # Each rank sends its data to rank 0
            if self.rank == 0:
                # Rank 0 collects from all
                all_data.append(np_data)
                for src in range(1, self.world_size):
                    received = self.receive(src, CommunicationType.NUMPY)
                    all_data.append(received)
                
                # Perform reduction
                if op == "sum":
                    result = np.sum(all_data, axis=0)
                elif op == "mean":
                    result = np.mean(all_data, axis=0)
                elif op == "max":
                    result = np.maximum.reduce(all_data)
                elif op == "min":
                    result = np.minimum.reduce(all_data)
                else:
                    raise ValueError(f"Unsupported operation: {op}")
                
                # Broadcast result
                for dest in range(1, self.world_size):
                    self.send(result, dest, CommunicationType.NUMPY)
            else:
                # Other ranks send to rank 0 and wait for result
                self.send(np_data, 0, CommunicationType.NUMPY)
                result = self.receive(0, CommunicationType.NUMPY)
            
            return mx.array(result)
        except Exception as e:
            logger.error(f"AllReduce failed: {e}")
            raise
    
    def barrier(self) -> None:
        """Synchronization barrier."""
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        barrier_id = f"barrier_{uuid.uuid4().hex}"
        
        # Create request
        request = distributed_comm_pb2.BarrierRequest(
            rank=self.rank,
            barrier_id=barrier_id
        )
        
        # Simple barrier implementation: all ranks check in with rank 0
        try:
            if self.rank == 0:
                # Rank 0 waits for all others
                checked_in = {0}  # Include self
                for src in range(1, self.world_size):
                    self.receive(src, CommunicationType.PICKLE)
                    checked_in.add(src)
                
                # Notify all others that barrier is complete
                for dest in range(1, self.world_size):
                    self.send("barrier_complete", dest, CommunicationType.PICKLE)
            else:
                # Other ranks check in with rank 0
                self.send(f"rank_{self.rank}_ready", 0, CommunicationType.PICKLE)
                # Wait for completion signal
                self.receive(0, CommunicationType.PICKLE)
            
            logger.debug(f"Rank {self.rank} passed barrier")
        except Exception as e:
            logger.error(f"Barrier failed: {e}")
            raise
    
    def send_tensor(self, tensor: mx.array, dest: int, compress: bool = True) -> None:
        """Optimized tensor sending with optional compression."""
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        if dest not in self._stubs:
            raise ValueError(f"No connection to rank {dest}")
        
        # Convert MLX tensor to numpy for serialization
        np_tensor = np.array(tensor)
        logger.debug(f"Sending tensor shape {np_tensor.shape} dtype {np_tensor.dtype} to rank {dest}")
        
        # Compress large tensors for faster transmission
        if compress and np_tensor.nbytes > 1024 * 1024:  # > 1MB
            import gzip
            compressed_data = gzip.compress(np_tensor.tobytes())
            metadata = {
                'compressed': 'gzip',
                'original_size': str(np_tensor.nbytes),
                'compression_ratio': f"{len(compressed_data) / np_tensor.nbytes:.2f}"
            }
            logger.debug(f"Compressed tensor from {np_tensor.nbytes} to {len(compressed_data)} bytes")
        else:
            compressed_data = np_tensor.tobytes()
            metadata = {'compressed': 'none'}
        
        # Create tensor data
        tensor_data = distributed_comm_pb2.TensorData(
            data=compressed_data,
            shape=list(np_tensor.shape),
            dtype=str(np_tensor.dtype)
        )
        
        comm_data = distributed_comm_pb2.CommData(
            comm_type="tensor",
            tensor_data=tensor_data,
            metadata=metadata
        )
        
        request = distributed_comm_pb2.SendRequest(
            source_rank=self.rank,
            dest_rank=dest,
            data=comm_data,
            tag="tensor"
        )
        
        try:
            response = self._stubs[dest].Send(request)
            if not response.success:
                raise RuntimeError(f"Tensor send failed: {response.message}")
            logger.debug(f"Successfully sent tensor to rank {dest}")
        except grpc.RpcError as e:
            logger.error(f"gRPC tensor send error: {e}")
            raise

    def receive_tensor(self, source: int, timeout: float = 30.0) -> mx.array:
        """Optimized tensor receiving with timeout."""
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        request = distributed_comm_pb2.ReceiveRequest(
            receiver_rank=self.rank,
            source_rank=source,
            tag="tensor",
            comm_type="tensor"
        )
        
        logger.debug(f"Waiting to receive tensor from rank {source}")
        
        # Use local servicer for receiving
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if hasattr(self, '_servicer'):
                    for response in self._servicer.Receive(request, None):
                        tensor_data = response.data.tensor_data
                        metadata = response.data.metadata
                        
                        # Decompress if needed
                        if metadata.get('compressed') == 'gzip':
                            import gzip
                            decompressed_data = gzip.decompress(tensor_data.data)
                            logger.debug(f"Decompressed tensor from {len(tensor_data.data)} bytes")
                        else:
                            decompressed_data = tensor_data.data
                        
                        # Reconstruct tensor
                        np_array = np.frombuffer(
                            decompressed_data, 
                            dtype=tensor_data.dtype
                        ).reshape(tensor_data.shape)
                        
                        result = mx.array(np_array)
                        logger.debug(f"Successfully received tensor shape {result.shape} from rank {source}")
                        return result
                else:
                    raise RuntimeError("Servicer not initialized")
            except Exception as e:
                logger.debug(f"Receive attempt failed: {e}")
                time.sleep(0.001)
        
        raise TimeoutError(f"Timeout waiting for tensor from rank {source}")

    def finalize(self) -> None:
        """Cleanup gRPC communication."""
        if self._initialized:
            if self._server:
                self._server.stop(0)
            for channel in self._channels.values():
                channel.close()
            self._initialized = False
            logger.info("gRPC communicator finalized")


def create_communicator(backend: CommunicationBackend = CommunicationBackend.GRPC) -> DistributedCommunicator:
    """Factory function to create appropriate communicator."""
    if backend == CommunicationBackend.GRPC:
        return GRPCCommunicator()
    else:
        raise ValueError(f"Unsupported backend: {backend}")


class CommunicationProfiler:
    """Profile communication performance."""
    
    def __init__(self):
        self.send_times = []
        self.receive_times = []
        self.broadcast_times = []
        self.allreduce_times = []
        self.barrier_times = []
    
    def profile_send(self, comm: DistributedCommunicator, *args, **kwargs):
        """Profile send operation."""
        start = time.time()
        result = comm.send(*args, **kwargs)
        self.send_times.append(time.time() - start)
        return result
    
    def profile_receive(self, comm: DistributedCommunicator, *args, **kwargs):
        """Profile receive operation."""
        start = time.time()
        result = comm.receive(*args, **kwargs)
        self.receive_times.append(time.time() - start)
        return result
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get communication statistics."""
        return {
            "send": {
                "count": len(self.send_times),
                "avg_time": np.mean(self.send_times) if self.send_times else 0,
                "total_time": sum(self.send_times)
            },
            "receive": {
                "count": len(self.receive_times),
                "avg_time": np.mean(self.receive_times) if self.receive_times else 0,
                "total_time": sum(self.receive_times)
            },
            "broadcast": {
                "count": len(self.broadcast_times),
                "avg_time": np.mean(self.broadcast_times) if self.broadcast_times else 0,
                "total_time": sum(self.broadcast_times)
            },
            "allreduce": {
                "count": len(self.allreduce_times),
                "avg_time": np.mean(self.allreduce_times) if self.allreduce_times else 0,
                "total_time": sum(self.allreduce_times)
            },
            "barrier": {
                "count": len(self.barrier_times),
                "avg_time": np.mean(self.barrier_times) if self.barrier_times else 0,
                "total_time": sum(self.barrier_times)
            }
        }