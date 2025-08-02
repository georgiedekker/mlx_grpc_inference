"""
Distributed System Integration Adapter

Provides clean integration with distributed inference systems
without modifying external code. Uses adapter pattern for loose coupling.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DistributedInferenceAdapter:
    """Adapter for distributed inference systems."""
    
    def __init__(self, fallback_to_local: bool = True):
        self.fallback_to_local = fallback_to_local
        self.distributed_available = self._check_distributed_availability()
        
        if self.distributed_available:
            try:
                # Try to import distributed components
                from mlx_distributed.api import DistributedInferenceAPI
                self.api = DistributedInferenceAPI()
                logger.info("Distributed integration initialized successfully")
            except ImportError:
                logger.warning("Distributed components not found, using fallback")
                self.distributed_available = False
                self.api = None
        else:
            self.api = None
    
    def _check_distributed_availability(self) -> bool:
        """Check if distributed system is available."""
        try:
            import mlx_distributed
            return True
        except ImportError:
            return False
    
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """Generate text using distributed inference or fallback."""
        if self.distributed_available and self.api:
            try:
                return self.api.generate(prompt, max_tokens=max_tokens, **kwargs)
            except Exception as e:
                logger.error(f"Distributed inference failed: {e}")
                if self.fallback_to_local:
                    return self._local_generate(prompt, max_tokens, **kwargs)
                raise
        elif self.fallback_to_local:
            return self._local_generate(prompt, max_tokens, **kwargs)
        else:
            raise RuntimeError("Distributed inference not available and fallback disabled")
    
    def _local_generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """Local fallback generation."""
        logger.info("Using local generation fallback")
        # In a real implementation, this would use local MLX models
        return f"[Local fallback] Response to: {prompt[:50]}..."


class DistributedCommunicationAdapter:
    """Adapter for distributed communication layer."""
    
    def __init__(self):
        self.distributed_available = self._check_availability()
        
        if self.distributed_available:
            try:
                from mlx_distributed.communication import DistributedComm
                self.comm = DistributedComm()
            except ImportError:
                self.distributed_available = False
                self.comm = None
    
    def _check_availability(self) -> bool:
        """Check if distributed communication layer is available."""
        try:
            import mlx_distributed.communication
            return True
        except ImportError:
            return False
    
    def broadcast(self, data: Any, source: int = 0) -> Any:
        """Broadcast data to all nodes."""
        if self.distributed_available and self.comm:
            return self.comm.broadcast(data, source)
        else:
            # Single node fallback
            return data
    
    def all_reduce(self, tensor: Any, op: str = "sum") -> Any:
        """All-reduce operation across nodes."""
        if self.distributed_available and self.comm:
            return self.comm.all_reduce(tensor, op)
        else:
            # Single node fallback
            return tensor


def get_integration_status() -> Dict[str, Any]:
    """Get comprehensive distributed system integration status."""
    status = {
        "distributed_available": False,
        "inference_available": False,
        "communication_available": False,
        "grpc_available": False,
        "api_endpoint": None,
        "version": None
    }
    
    # Check if distributed package is installed
    try:
        import mlx_distributed
        status["distributed_available"] = True
        status["version"] = getattr(mlx_distributed, "__version__", "unknown")
    except ImportError:
        return status
    
    # Check specific components
    try:
        from mlx_distributed.api import DistributedInferenceAPI
        status["inference_available"] = True
    except ImportError:
        pass
    
    try:
        from mlx_distributed.communication import DistributedComm
        status["communication_available"] = True
    except ImportError:
        pass
    
    # Check if distributed API is running
    try:
        import requests
        response = requests.get("http://localhost:8100/health", timeout=2)
        if response.status_code == 200:
            status["api_endpoint"] = "http://localhost:8100"
            status["grpc_available"] = True
    except:
        pass
    
    return status


def is_distributed_available() -> bool:
    """Quick check if distributed integration is available."""
    return get_integration_status()["distributed_available"]


def create_inference_adapter(fallback_to_local: bool = True) -> DistributedInferenceAdapter:
    """Create a distributed inference adapter."""
    return DistributedInferenceAdapter(fallback_to_local=fallback_to_local)


def create_communication_adapter() -> DistributedCommunicationAdapter:
    """Create a distributed communication adapter."""
    return DistributedCommunicationAdapter()


# Configuration mapping for distributed systems
class ConfigurationAdapter:
    """Maps training configs to distributed system configs."""
    
    @staticmethod
    def map_to_distributed_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert training config to distributed system format."""
        return {
            "model": {
                "name": training_config.get("model", {}).get("name"),
                "dtype": training_config.get("model", {}).get("dtype", "float16"),
            },
            "distributed": {
                "enabled": training_config.get("distributed", {}).get("enabled", False),
                "world_size": training_config.get("distributed", {}).get("nodes", 1),
                "strategy": "pipeline_parallel",  # Distributed strategy
            },
            "inference": {
                "max_batch_size": training_config.get("training", {}).get("batch_size", 1),
                "max_sequence_length": training_config.get("data", {}).get("max_seq_length", 2048),
            }
        }
    
    @staticmethod
    def map_from_distributed_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert distributed system response to local format."""
        return {
            "generated_text": response.get("text", ""),
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "latency_ms": response.get("latency", 0),
            "model_used": response.get("model", "unknown"),
        }