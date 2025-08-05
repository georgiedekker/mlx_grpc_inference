#!/usr/bin/env python3
"""
Hybrid Distributed Inference System
Combines PyTorch distributed, gRPC, and direct TCP for optimal performance
"""
import os
import torch
import torch.distributed as dist
import numpy as np
import socket
import struct
import logging
import asyncio
from typing import Optional, Tuple, Dict, Any
import grpc
from concurrent import futures

# Import our protobuf definitions
from src.communication import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)

class DirectTCPTransfer:
    """Direct TCP socket transfer for bulk tensor data"""
    
    def __init__(self, port_base: int = 30000):
        self.port_base = port_base
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
    async def send_tensor(self, tensor: torch.Tensor, dest_rank: int) -> None:
        """Send tensor directly over TCP socket"""
        # Convert to numpy for efficient transfer
        numpy_array = tensor.cpu().numpy()
        
        # Prepare metadata
        dtype_str = str(numpy_array.dtype)
        shape = numpy_array.shape
        
        # Connect to destination
        dest_ip = self._get_ip_for_rank(dest_rank)
        dest_port = self.port_base + dest_rank
        
        reader, writer = await asyncio.open_connection(dest_ip, dest_port)
        
        try:
            # Send metadata
            metadata = {
                'dtype': dtype_str,
                'shape': shape,
                'size': numpy_array.nbytes
            }
            metadata_bytes = str(metadata).encode()
            writer.write(struct.pack('I', len(metadata_bytes)))
            writer.write(metadata_bytes)
            
            # Send tensor data
            writer.write(numpy_array.tobytes())
            await writer.drain()
            
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def receive_tensor(self) -> torch.Tensor:
        """Receive tensor over TCP socket"""
        port = self.port_base + self.rank
        
        server = await asyncio.start_server(
            self._handle_client, '0.0.0.0', port
        )
        
        async with server:
            await server.serve_forever()
    
    async def _handle_client(self, reader, writer):
        """Handle incoming tensor transfer"""
        # Read metadata size
        size_data = await reader.read(4)
        metadata_size = struct.unpack('I', size_data)[0]
        
        # Read metadata
        metadata_bytes = await reader.read(metadata_size)
        metadata = eval(metadata_bytes.decode())
        
        # Read tensor data
        tensor_bytes = await reader.read(metadata['size'])
        
        # Reconstruct tensor
        numpy_array = np.frombuffer(tensor_bytes, dtype=metadata['dtype'])
        numpy_array = numpy_array.reshape(metadata['shape'])
        tensor = torch.from_numpy(numpy_array)
        
        writer.close()
        await writer.wait_closed()
        
        return tensor
    
    def _get_ip_for_rank(self, rank: int) -> str:
        """Get IP address for a given rank"""
        if rank == 0:
            return "192.168.5.1"  # mini1
        elif rank == 1:
            return "192.168.5.2"  # mini2
        else:
            raise ValueError(f"Unknown rank: {rank}")


class GRPCTensorTransfer(inference_pb2_grpc.InferenceServicer):
    """gRPC-based tensor transfer for reliability"""
    
    def __init__(self, port: int = 50051):
        self.port = port
        self.rank = int(os.environ.get('RANK', 0))
        self.received_tensors = {}
        
    def TransferTensor(self, request, context):
        """Handle incoming tensor transfer via gRPC"""
        # Deserialize tensor
        tensor_data = np.frombuffer(request.data, dtype=np.float32)
        tensor_data = tensor_data.reshape(request.shape)
        tensor = torch.from_numpy(tensor_data)
        
        # Store for retrieval
        self.received_tensors[request.tensor_id] = tensor
        
        return inference_pb2.TransferResponse(
            success=True,
            message=f"Received tensor {request.tensor_id}"
        )
    
    async def send_tensor(self, tensor: torch.Tensor, dest_rank: int, tensor_id: str):
        """Send tensor via gRPC"""
        dest_ip = self._get_ip_for_rank(dest_rank)
        channel = grpc.aio.insecure_channel(f'{dest_ip}:{self.port + dest_rank}')
        stub = inference_pb2_grpc.InferenceStub(channel)
        
        # Prepare tensor for transfer
        numpy_array = tensor.cpu().numpy().astype(np.float32)
        
        request = inference_pb2.TensorTransfer(
            tensor_id=tensor_id,
            shape=list(numpy_array.shape),
            data=numpy_array.tobytes()
        )
        
        response = await stub.TransferTensor(request)
        await channel.close()
        
        return response.success
    
    def _get_ip_for_rank(self, rank: int) -> str:
        """Get IP address for a given rank"""
        if rank == 0:
            return "192.168.5.1"
        elif rank == 1:
            return "192.168.5.2"
        else:
            raise ValueError(f"Unknown rank: {rank}")


class HybridDistributedInference:
    """
    Hybrid distributed inference system combining:
    1. PyTorch distributed for coordination and small tensors
    2. gRPC for reliable medium-size tensor transfer
    3. Direct TCP for bulk tensor transfer
    """
    
    def __init__(self, 
                 small_tensor_threshold: int = 1024 * 1024,  # 1MB
                 large_tensor_threshold: int = 10 * 1024 * 1024):  # 10MB
        
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.small_threshold = small_tensor_threshold
        self.large_threshold = large_tensor_threshold
        
        # Initialize PyTorch distributed
        if self.world_size > 1:
            self._init_pytorch_distributed()
        
        # Initialize transfer mechanisms
        self.tcp_transfer = DirectTCPTransfer()
        self.grpc_transfer = GRPCTensorTransfer()
        
        # Start gRPC server
        self._start_grpc_server()
        
        logger.info(f"Initialized hybrid distributed system on rank {self.rank}")
    
    def _init_pytorch_distributed(self):
        """Initialize PyTorch distributed with Gloo backend"""
        if not dist.is_initialized():
            # Ensure we're using the Thunderbolt interface
            os.environ['GLOO_SOCKET_IFNAME'] = 'bridge0'
            
            dist.init_process_group(
                backend='gloo',
                init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                rank=self.rank,
                world_size=self.world_size
            )
            logger.info("PyTorch distributed initialized with Gloo backend")
    
    def _start_grpc_server(self):
        """Start gRPC server for tensor transfers"""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        inference_pb2_grpc.add_InferenceServicer_to_server(
            self.grpc_transfer, server
        )
        port = 50051 + self.rank
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        logger.info(f"gRPC server started on port {port}")
    
    async def transfer_tensor(self, tensor: torch.Tensor, dest_rank: int) -> None:
        """
        Transfer tensor to destination rank using optimal method
        """
        tensor_size = tensor.numel() * tensor.element_size()
        
        if tensor_size < self.small_threshold:
            # Use PyTorch distributed for small tensors
            await self._pytorch_transfer(tensor, dest_rank)
        elif tensor_size < self.large_threshold:
            # Use gRPC for medium tensors
            await self._grpc_transfer(tensor, dest_rank)
        else:
            # Use direct TCP for large tensors
            await self._tcp_transfer(tensor, dest_rank)
    
    async def _pytorch_transfer(self, tensor: torch.Tensor, dest_rank: int):
        """Transfer using PyTorch distributed"""
        if self.rank == dest_rank:
            return tensor
        
        # Use point-to-point communication
        if self.rank == 0:  # Sender
            dist.send(tensor, dst=dest_rank)
        else:  # Receiver
            dist.recv(tensor, src=0)
        
        logger.debug(f"Transferred tensor via PyTorch distributed")
    
    async def _grpc_transfer(self, tensor: torch.Tensor, dest_rank: int):
        """Transfer using gRPC"""
        tensor_id = f"tensor_{self.rank}_{dest_rank}_{id(tensor)}"
        success = await self.grpc_transfer.send_tensor(tensor, dest_rank, tensor_id)
        logger.debug(f"Transferred tensor via gRPC: {success}")
    
    async def _tcp_transfer(self, tensor: torch.Tensor, dest_rank: int):
        """Transfer using direct TCP"""
        await self.tcp_transfer.send_tensor(tensor, dest_rank)
        logger.debug(f"Transferred tensor via direct TCP")
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all ranks"""
        if self.world_size == 1:
            return tensor
        
        dist.broadcast(tensor, src=src)
        return tensor
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce operation for gradient aggregation"""
        if self.world_size == 1:
            return tensor
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor
    
    def barrier(self):
        """Synchronization barrier"""
        if self.world_size > 1:
            dist.barrier()
    
    def cleanup(self):
        """Clean up distributed resources"""
        if dist.is_initialized():
            dist.destroy_process_group()