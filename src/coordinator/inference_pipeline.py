"""
Distributed inference pipeline for coordinating execution across devices.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
import mlx.core as mx

from ..core.config import ClusterConfig
from ..model.inference import LayerProcessor
from ..communication.connection_pool import ConnectionPool

logger = logging.getLogger(__name__)


class DistributedInferencePipeline:
    """Manages distributed forward pass execution across devices."""
    
    def __init__(self, 
                 config: ClusterConfig,
                 layer_processor: LayerProcessor,
                 connection_pool: ConnectionPool):
        """
        Initialize distributed inference pipeline.
        
        Args:
            config: Cluster configuration
            layer_processor: Local layer processor
            connection_pool: Pool of gRPC connections
        """
        self.config = config
        self.layer_processor = layer_processor
        self.connection_pool = connection_pool
        self.device_config = config.get_local_device()
        
        logger.info(f"DistributedInferencePipeline initialized for {self.device_config.device_id}")
    
    async def forward_pass(self, 
                          input_tensor: mx.array, 
                          request_id: str) -> Tuple[mx.array, Dict[str, float]]:
        """
        Execute distributed forward pass across all devices.
        
        Args:
            input_tensor: Input tensor (token IDs or hidden states)
            request_id: Unique request identifier for tracking
            
        Returns:
            Tuple of (final_hidden_states, device_timing_info)
        """
        device_times = {}
        hidden_states = input_tensor
        
        # Process embeddings if this is the first device in pipeline
        if self._is_embedding_device():
            hidden_states = await self._process_embeddings(input_tensor, device_times)
        
        # Process local layers
        hidden_states = await self._process_local_layers(hidden_states, device_times)
        
        # Process through worker devices in pipeline order
        hidden_states = await self._process_worker_devices(
            hidden_states, request_id, device_times
        )
        
        return hidden_states, device_times
    
    async def _process_embeddings(self, 
                                 input_ids: mx.array, 
                                 device_times: Dict[str, float]) -> mx.array:
        """Process input embeddings on the coordinator device."""
        start = time.time()
        hidden_states = self.layer_processor.process_embedding(input_ids)
        mx.eval(hidden_states)
        device_times['embedding'] = (time.time() - start) * 1000
        
        logger.debug(f"Processed embeddings in {device_times['embedding']:.1f}ms")
        return hidden_states
    
    async def _process_local_layers(self, 
                                   hidden_states: mx.array,
                                   device_times: Dict[str, float]) -> mx.array:
        """Process layers assigned to the local device."""
        start = time.time()
        
        local_layers = self.config.model.get_device_layers(self.device_config.device_id)
        if local_layers:
            hidden_states = self.layer_processor.process(
                hidden_states,
                local_layers,
                {}  # Context can be extended for attention masks, etc.
            )
            mx.eval(hidden_states)
            
        device_times[self.device_config.device_id] = (time.time() - start) * 1000
        logger.debug(f"Processed local layers {local_layers} in {device_times[self.device_config.device_id]:.1f}ms")
        
        return hidden_states
    
    async def _process_worker_devices(self, 
                                     hidden_states: mx.array,
                                     request_id: str,
                                     device_times: Dict[str, float]) -> mx.array:
        """Process through worker devices in pipeline order."""
        current_device_id = self.device_config.device_id
        
        while True:
            # Get next device in pipeline
            next_client = self.connection_pool.get_next_device_client(current_device_id)
            if not next_client:
                break
            
            next_device = self.config.get_device_by_rank(
                self.config.get_device(current_device_id).rank + 1
            )
            
            # Get layers assigned to next device
            next_layers = self.config.model.get_device_layers(next_device.device_id)
            if not next_layers:
                logger.warning(f"No layers assigned to {next_device.device_id}")
                break
            
            # Process on remote device
            hidden_states = await self._process_remote_device(
                next_client,
                hidden_states,
                next_layers,
                request_id,
                next_device.device_id,
                device_times
            )
            
            current_device_id = next_device.device_id
        
        # Validate pipeline completion
        self._validate_pipeline_completion(current_device_id)
        
        return hidden_states
    
    async def _process_remote_device(self,
                                    client,
                                    hidden_states: mx.array,
                                    layer_indices: List[int],
                                    request_id: str,
                                    device_id: str,
                                    device_times: Dict[str, float]) -> mx.array:
        """Process layers on a remote device via gRPC."""
        logger.debug(f"Sending to {device_id} for layers {layer_indices}")
        
        start = time.time()
        result = client.process_layers(
            hidden_states,
            layer_indices,
            request_id,
            {}  # Context for attention masks, etc.
        )
        
        # Record timing from remote device
        device_times[device_id] = result.processing_time_ms
        
        # Verify result validity
        if result.output_tensor is None:
            raise RuntimeError(f"No output received from device {device_id}")
        
        return result.output_tensor
    
    def _is_embedding_device(self) -> bool:
        """Check if this device should process embeddings."""
        # Typically the coordinator or first device processes embeddings
        return self.device_config.rank == 0
    
    def _validate_pipeline_completion(self, final_device_id: str):
        """Validate that the pipeline completed on the expected device."""
        expected_last_device = self._get_last_device_id()
        if final_device_id != expected_last_device:
            logger.warning(
                f"Pipeline ended at {final_device_id}, expected {expected_last_device}"
            )
    
    def _get_last_device_id(self) -> str:
        """Get the ID of the device processing the last layers."""
        max_layer = -1
        last_device_id = None
        
        for device_id, layers in self.config.model.layer_distribution.items():
            if layers and max(layers) > max_layer:
                max_layer = max(layers)
                last_device_id = device_id
        
        return last_device_id
    
    def get_pipeline_info(self) -> Dict[str, any]:
        """Get information about the pipeline configuration."""
        devices = []
        for device in self.config.get_devices_by_rank():
            layers = self.config.model.get_device_layers(device.device_id)
            devices.append({
                'device_id': device.device_id,
                'rank': device.rank,
                'layers': layers,
                'layer_count': len(layers) if layers else 0
            })
        
        return {
            'total_devices': len(self.config.devices),
            'total_layers': sum(len(layers) for layers in self.config.model.layer_distribution.values()),
            'device_pipeline': devices,
            'is_coordinator': self.device_config.rank == 0
        }