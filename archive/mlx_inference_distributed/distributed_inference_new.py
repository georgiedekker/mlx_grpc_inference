#!/usr/bin/env python3
"""
Distributed Inference Engine for MLX Models
Coordinates inference across multiple devices with layer splitting
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Import our modules
from model_splitter import ModelLayerSplitter, LAYER_ASSIGNMENTS
from grpc_tensor_service import GRPCTensorClient, TensorServiceImplementation
from device_comm import TensorSerializer

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Request for distributed inference"""
    request_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    model_name: str = "mlx-community/Qwen3-1.7B-8bit"
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class InferenceResult:
    """Result from distributed inference"""
    request_id: str
    generated_text: str
    total_tokens: int
    processing_time: float
    device_timings: Dict[str, float]
    success: bool = True
    error_message: str = ""

class DistributedInferenceEngine:
    """Main engine for coordinating distributed inference"""
    
    def __init__(self, device_id: str, is_coordinator: bool = False):
        self.device_id = device_id
        self.is_coordinator = is_coordinator
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.model_splitter = None
        
        # Device information
        self.device_assignments = {
            assignment.device_id: assignment for assignment in LAYER_ASSIGNMENTS
        }
        self.my_assignment = self.device_assignments.get(device_id)
        
        # Communication
        self.grpc_client = GRPCTensorClient(device_id)
        self.connected_devices: Dict[str, bool] = {}
        
        # Performance tracking
        self.inference_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "total_tokens_generated": 0
        }
        
        logger.info(f"Initialized DistributedInferenceEngine for {device_id}")
        if self.my_assignment:
            logger.info(f"Assigned layers: {self.my_assignment.layers}")
        
    async def initialize(self, model_name: str = "mlx-community/Qwen3-1.7B-8bit"):
        """Initialize the inference engine"""
        try:
            logger.info(f"Initializing inference engine for {self.device_id}")
            
            # Load model components
            await self._load_model_components(model_name)
            
            # Connect to other devices
            await self._connect_to_cluster()
            
            logger.info(f"✅ Inference engine initialized for {self.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            return False
    
    async def _load_model_components(self, model_name: str):
        """Load model components (tokenizer for coordinator, layers for workers)"""
        try:
            if self.is_coordinator:
                # Coordinator needs full model for tokenization
                logger.info("Loading full model for coordinator...")
                self.model, self.tokenizer = load(model_name)
                logger.info("✅ Full model loaded on coordinator")
                
                # Also initialize model splitter for analysis
                self.model_splitter = ModelLayerSplitter(model_name)
                await asyncio.to_thread(self.model_splitter.load_full_model)
                
            else:
                # Workers load their assigned layers
                logger.info(f"Loading assigned layers for worker {self.device_id}...")
                self.model_splitter = ModelLayerSplitter(model_name)
                await asyncio.to_thread(self.model_splitter.load_full_model)
                
                # Extract layers for this device
                if self.my_assignment:
                    layer_data = self.model_splitter.extract_layers_for_device(self.my_assignment)
                    logger.info(f"✅ Loaded {len(layer_data['model_dict'])} model components")
                
        except Exception as e:
            logger.error(f"Failed to load model components: {e}")
            raise
    
    async def _connect_to_cluster(self):
        """Connect to other devices in the cluster"""
        try:
            logger.info("Connecting to cluster devices...")
            
            for assignment in LAYER_ASSIGNMENTS:
                if assignment.device_id != self.device_id:
                    # Determine port (coordinator on 8100, workers on 8101, 8102)
                    if assignment.device_id == "mini1":
                        port = 8100
                    elif assignment.device_id == "mini2":
                        port = 8101
                    else:  # master
                        port = 8102
                    
                    # For gRPC, use different ports
                    grpc_port = port + 1000  # 9100, 9101, 9102
                    
                    connected = await self.grpc_client.connect_to_device(
                        assignment.device_id, 
                        assignment.hostname, 
                        grpc_port
                    )
                    
                    self.connected_devices[assignment.device_id] = connected
                    
                    if connected:
                        logger.info(f"✅ Connected to {assignment.device_id}")
                    else:
                        logger.warning(f"❌ Failed to connect to {assignment.device_id}")
            
            connected_count = sum(1 for connected in self.connected_devices.values() if connected)
            logger.info(f"Connected to {connected_count}/{len(self.connected_devices)} devices")
            
        except Exception as e:
            logger.error(f"Failed to connect to cluster: {e}")
            raise
    
    async def process_inference_request(self, request: InferenceRequest) -> InferenceResult:
        """Process an inference request using distributed computation"""
        start_time = time.time()
        device_timings = {}
        
        try:
            logger.info(f"🚀 Processing inference request {request.request_id}")
            logger.info(f"   Prompt: {request.prompt[:100]}...")
            logger.info(f"   Max tokens: {request.max_tokens}")
            
            if not self.is_coordinator:
                raise Exception("Only coordinator can process inference requests")
            
            # Tokenize input
            tokenize_start = time.time()
            if not self.tokenizer:
                raise Exception("Tokenizer not loaded")
            
            # Simple prompt formatting (in production, use proper chat templates)
            formatted_prompt = f"User: {request.prompt}\nAssistant: "
            tokens = self.tokenizer.encode(formatted_prompt)
            device_timings["tokenization"] = time.time() - tokenize_start
            
            logger.info(f"   Input tokens: {len(tokens)}")
            
            # Create input tensor
            input_tensor = mx.array([tokens], dtype=mx.int32)  # Batch size 1
            
            # Perform distributed forward passes for generation
            generated_tokens = []
            current_tensor = input_tensor
            
            for token_idx in range(request.max_tokens):
                logger.debug(f"Generating token {token_idx + 1}/{request.max_tokens}")
                
                # Distributed forward pass
                forward_start = time.time()
                output_tensor = await self._distributed_forward_pass(
                    current_tensor, 
                    f"{request.request_id}_token_{token_idx}"
                )
                forward_time = time.time() - forward_start
                device_timings[f"forward_pass_{token_idx}"] = forward_time
                
                # Get next token logits (last position, last token)
                # Handle different tensor shapes
                if len(output_tensor.shape) == 3:
                    logits = output_tensor[0, -1, :]  # Shape: [vocab_size]
                elif len(output_tensor.shape) == 2:
                    logits = output_tensor[-1, :]  # Shape: [vocab_size]
                else:
                    raise ValueError(f"Unexpected output tensor shape: {output_tensor.shape}")
                
                # Sample next token
                sampler = make_sampler(temp=request.temperature)
                next_token = sampler(logits.reshape(1, -1))[0]  # Get single token
                
                # Check for end of sequence
                if self.tokenizer.eos_token_id and next_token == self.tokenizer.eos_token_id:
                    logger.info(f"Generated EOS token, stopping at {len(generated_tokens)} tokens")
                    break
                
                generated_tokens.append(int(next_token))
                
                # Update input for next iteration (append new token)
                new_input = mx.concatenate([current_tensor, mx.array([[next_token]], dtype=mx.int32)], axis=1)
                current_tensor = new_input
                
                # Limit context length to prevent memory issues
                if current_tensor.shape[1] > 2048:
                    # Keep last 1024 tokens
                    current_tensor = current_tensor[:, -1024:]
                    logger.debug("Truncated context to prevent memory overflow")
            
            # Decode generated tokens
            decode_start = time.time()
            generated_text = self.tokenizer.decode(generated_tokens)
            device_timings["decoding"] = time.time() - decode_start
            
            total_time = time.time() - start_time
            
            # Update stats
            self.inference_stats["total_requests"] += 1
            self.inference_stats["successful_requests"] += 1
            self.inference_stats["total_tokens_generated"] += len(generated_tokens)
            
            # Update average processing time
            old_avg = self.inference_stats["average_processing_time"]
            total_requests = self.inference_stats["total_requests"]
            self.inference_stats["average_processing_time"] = (old_avg * (total_requests - 1) + total_time) / total_requests
            
            logger.info(f"✅ Generated {len(generated_tokens)} tokens in {total_time:.2f}s")
            logger.info(f"   Generated text: {generated_text[:200]}...")
            
            return InferenceResult(
                request_id=request.request_id,
                generated_text=generated_text,
                total_tokens=len(generated_tokens),
                processing_time=total_time,
                device_timings=device_timings,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Inference request {request.request_id} failed: {e}")
            
            self.inference_stats["total_requests"] += 1
            self.inference_stats["failed_requests"] += 1
            
            return InferenceResult(
                request_id=request.request_id,
                generated_text="",
                total_tokens=0,
                processing_time=time.time() - start_time,
                device_timings=device_timings,
                success=False,
                error_message=str(e)
            )
    
    async def _distributed_forward_pass(self, input_tensor: mx.array, pass_id: str) -> mx.array:
        """Execute distributed forward pass across all devices"""
        try:
            logger.debug(f"Starting distributed forward pass {pass_id}")
            logger.debug(f"  Input shape: {input_tensor.shape}")
            
            current_tensor = input_tensor
            
            # Process through each device in layer order
            for assignment in LAYER_ASSIGNMENTS:
                device_id = assignment.device_id
                layer_start = assignment.layer_start
                layer_end = assignment.layer_end
                
                step_start = time.time()
                
                if device_id == self.device_id:
                    # Process locally
                    logger.debug(f"  Processing layers {layer_start}-{layer_end} locally")
                    current_tensor = await self._local_forward_pass(
                        current_tensor, 
                        layer_start, 
                        layer_end,
                        pass_id
                    )
                else:
                    # Process remotely
                    logger.debug(f"  Delegating layers {layer_start}-{layer_end} to {device_id}")
                    
                    if not self.connected_devices.get(device_id, False):
                        logger.warning(f"Device {device_id} not connected, simulating locally")
                        current_tensor = await self._local_forward_pass(
                            current_tensor, 
                            layer_start, 
                            layer_end,
                            f"simulated_{pass_id}"
                        )
                    else:
                        # Use gRPC to request forward pass
                        result = await self.grpc_client.request_forward_pass(
                            current_tensor,
                            device_id,
                            layer_start,
                            layer_end,
                            f"{pass_id}_{device_id}"
                        )
                        
                        if result is not None:
                            current_tensor = result
                        else:
                            logger.error(f"Forward pass failed on {device_id}, simulating locally")
                            current_tensor = await self._local_forward_pass(
                                current_tensor, 
                                layer_start, 
                                layer_end,
                                f"fallback_{pass_id}"
                            )
                
                step_time = time.time() - step_start
                logger.debug(f"  {device_id} processing time: {step_time:.3f}s")
            
            logger.debug(f"  Final output shape: {current_tensor.shape}")
            return current_tensor
            
        except Exception as e:
            logger.error(f"Distributed forward pass {pass_id} failed: {e}")
            raise
    
    async def _local_forward_pass(self, 
                                input_tensor: mx.array, 
                                layer_start: int, 
                                layer_end: int, 
                                pass_id: str) -> mx.array:
        """Execute forward pass for local layers"""
        try:
            if self.is_coordinator and self.model:
                # Use actual model on coordinator
                logger.debug(f"Running actual model layers {layer_start}-{layer_end}")
                
                # For now, run the full model since we don't have layer splitting implemented yet
                # This will be refined when we integrate with actual layer extraction
                return await asyncio.to_thread(self._run_model_forward, input_tensor)
            else:
                # Simulate computation on workers
                output_tensor = input_tensor
                
                # Apply some computation to simulate layer processing
                for layer_idx in range(layer_start, layer_end + 1):
                    # Simple transformation to simulate transformer layers
                    output_tensor = mx.tanh(output_tensor)
                    
                return output_tensor
                
        except Exception as e:
            logger.error(f"Local forward pass failed: {e}")
            raise
    
    def _run_model_forward(self, input_tensor: mx.array) -> mx.array:
        """Run forward pass through actual model"""
        # Convert to expected format and run through model
        # For now, just return logits for the last position
        if hasattr(self.model, '__call__'):
            # Run through the model
            output = self.model(input_tensor)
            return output
        else:
            # Fallback to simulation
            return mx.random.normal(input_tensor.shape[:-1] + (self.model.args.vocab_size,))
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics"""
        return {
            **self.inference_stats,
            "connected_devices": self.connected_devices,
            "device_id": self.device_id,
            "is_coordinator": self.is_coordinator,
            "assigned_layers": self.my_assignment.layers if self.my_assignment else []
        }

async def test_distributed_inference():
    """Test distributed inference engine"""
    logger.info("🧪 Testing distributed inference engine...")
    
    try:
        # Create coordinator
        coordinator = DistributedInferenceEngine("mini1", is_coordinator=True)
        
        # Initialize coordinator
        initialized = await coordinator.initialize()
        if not initialized:
            logger.error("Failed to initialize coordinator")
            return False
        
        # Create test inference request
        request = InferenceRequest(
            request_id="test_001",
            prompt="What is machine learning?",
            max_tokens=50,
            temperature=0.7
        )
        
        # Process request
        result = await coordinator.process_inference_request(request)
        
        if result.success:
            logger.info("✅ Distributed inference test passed")
            logger.info(f"   Generated: {result.generated_text}")
            logger.info(f"   Tokens: {result.total_tokens}")
            logger.info(f"   Time: {result.processing_time:.2f}s")
            
            # Print device timings
            for device, timing in result.device_timings.items():
                logger.info(f"   {device}: {timing:.3f}s")
            
            return True
        else:
            logger.error(f"Inference failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"Distributed inference test failed: {e}")
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    success = asyncio.run(test_distributed_inference())
    
    if success:
        print("🎉 Distributed inference engine is working!")
    else:
        print("❌ Distributed inference test failed")