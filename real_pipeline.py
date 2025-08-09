#!/usr/bin/env python3
"""
REAL pipeline parallelism that ACTUALLY makes mini2 process layers.
This implementation properly passes activations between GPUs.
"""
import os
import time
import logging
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model, load_tokenizer
# KVCache import removed - not needed for basic test
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Rank %(rank)s] - %(message)s')
logger = logging.getLogger(__name__)

# Force GPU
mx.set_default_device(mx.gpu)

class RealPipelineModel:
    """
    Actually implements pipeline parallelism with activation passing.
    Mini1 (rank 0): processes layers 0-13
    Mini2 (rank 1): processes layers 14-27
    """
    
    def __init__(self, model_name="mlx-community/Qwen3-1.7B-8bit"):
        # Initialize distributed
        self.group = mx.distributed.init()
        if not self.group:
            raise RuntimeError("Failed to initialize distributed")
        
        self.rank = self.group.rank()
        self.world_size = self.group.size()
        
        # Add rank to logger
        logging.LoggerAdapter(logger, {'rank': self.rank})
        
        logger.info(f"Initializing on {os.uname().nodename} as rank {self.rank}/{self.world_size}")
        
        # Load model and tokenizer
        from pathlib import Path
        from huggingface_hub import snapshot_download
        
        # Download model files
        model_path = Path(snapshot_download(
            model_name,
            allow_patterns=["*.json", "*.py", "tokenizer.model", "*.tiktoken", "*.txt", "*.safetensors"]
        ))
        
        # Load model and tokenizer
        self.model, self.config = load_model(model_path)
        self.tokenizer = load_tokenizer(model_path, {"trust_remote_code": True})
        
        # Setup pipeline sharding
        self._setup_pipeline()
        
        # Log memory
        memory = mx.get_active_memory() / (1024**3)
        logger.info(f"GPU memory after setup: {memory:.2f} GB on {os.uname().nodename}")
    
    def _setup_pipeline(self):
        """Shard the model layers across GPUs."""
        # Get the actual model (unwrap if needed)
        if hasattr(self.model, 'model'):
            actual_model = self.model.model
        else:
            actual_model = self.model
        
        # Find layers
        if hasattr(actual_model, 'layers'):
            self.all_layers = actual_model.layers
        elif hasattr(actual_model, 'blocks'):
            self.all_layers = actual_model.blocks
        else:
            raise ValueError("Cannot find layers")
        
        self.num_layers = len(self.all_layers)
        layers_per_rank = self.num_layers // self.world_size
        
        # Calculate layer assignment
        self.start_layer = self.rank * layers_per_rank
        self.end_layer = (self.rank + 1) * layers_per_rank if self.rank < self.world_size - 1 else self.num_layers
        
        logger.info(f"Will process layers {self.start_layer} to {self.end_layer-1} ({self.end_layer - self.start_layer} layers)")
        
        # Store components based on rank
        self.embed_tokens = actual_model.embed_tokens if self.rank == 0 else None
        self.norm = actual_model.norm if self.rank == self.world_size - 1 else None
        
        # Get vocabulary size for final projection
        if hasattr(self.model, 'lm_head'):
            self.lm_head = self.model.lm_head if self.rank == self.world_size - 1 else None
        else:
            self.lm_head = None
    
    def forward_pipeline(self, input_ids):
        """
        Pipeline forward pass with REAL activation passing between GPUs.
        """
        logger.info(f"Starting forward pass on {os.uname().nodename}")
        
        # Rank 0: Process embedding and first layers
        if self.rank == 0:
            # Embed tokens
            h = self.embed_tokens(input_ids)
            logger.info(f"Embedded tokens, shape: {h.shape}")
            
            # Process layers 0-13
            for i in range(self.start_layer, self.end_layer):
                h = self.all_layers[i](h)
                if i % 5 == 0:  # Log every 5 layers
                    logger.info(f"Processed layer {i}")
            
            # Send activations to rank 1
            logger.info(f"Sending activations to rank 1, shape: {h.shape}")
            self._send_activations(h, dest=1)
            
            # Wait for final output from rank 1
            logger.info("Waiting for final output from rank 1...")
            output = self._recv_activations(src=1)
            logger.info(f"Received final output from rank 1")
            return output
        
        # Rank 1: Process remaining layers
        elif self.rank == 1:
            # Receive activations from rank 0
            logger.info("Waiting for activations from rank 0...")
            h = self._recv_activations(src=0)
            logger.info(f"Received activations from rank 0, shape: {h.shape}")
            
            # Process layers 14-27
            for i in range(self.start_layer, self.end_layer):
                h = self.all_layers[i](h)
                if i % 5 == 0:  # Log every 5 layers
                    logger.info(f"Processed layer {i}")
            
            # Apply final norm
            if self.norm:
                h = self.norm(h)
                logger.info("Applied final norm")
            
            # Apply language model head
            if self.lm_head:
                output = self.lm_head(h)
                logger.info(f"Applied lm_head, output shape: {output.shape}")
            else:
                output = h
            
            # Send final output back to rank 0
            logger.info("Sending final output back to rank 0")
            self._send_activations(output, dest=0)
            
            return output
    
    def _send_activations(self, tensor, dest):
        """Send tensor to destination rank."""
        # Flatten for sending
        shape = tensor.shape
        flat = tensor.reshape(-1)
        
        # Send shape first (as int32)
        shape_array = mx.array(shape, dtype=mx.int32)
        mx.distributed.send(shape_array, dst=dest, group=self.group)
        
        # Send data
        mx.distributed.send(flat, dst=dest, group=self.group)
        mx.eval(flat)  # Ensure send completes
        
        logger.info(f"Sent tensor shape {shape} to rank {dest}")
    
    def _recv_activations(self, src):
        """Receive tensor from source rank."""
        # Receive shape first
        shape_array = mx.zeros((3,), dtype=mx.int32)
        mx.distributed.recv(shape_array, src=src, group=self.group)
        mx.eval(shape_array)
        
        # Extract shape
        shape = tuple(int(s) for s in shape_array if s > 0)
        
        # Calculate total elements
        num_elements = 1
        for s in shape:
            num_elements *= s
        
        # Receive data
        flat = mx.zeros((num_elements,), dtype=mx.float32)
        mx.distributed.recv(flat, src=src, group=self.group)
        mx.eval(flat)
        
        # Reshape
        tensor = flat.reshape(shape)
        logger.info(f"Received tensor shape {shape} from rank {src}")
        
        return tensor
    
    def generate(self, prompt, max_tokens=20):
        """Generate text using pipeline parallelism."""
        # Tokenize prompt
        input_ids = mx.array(self.tokenizer.encode(prompt))
        
        logger.info(f"Generating with prompt: '{prompt[:50]}...'")
        
        generated_tokens = []
        
        for _ in range(max_tokens):
            # Forward pass through pipeline
            logits = self.forward_pipeline(input_ids[None])  # Add batch dimension
            
            # Get next token (only rank 0 decides)
            if self.rank == 0:
                # Take last token's logits
                next_token_logits = logits[0, -1, :]
                
                # Simple argmax sampling
                next_token = mx.argmax(next_token_logits)
                next_token = int(next_token.item())
                generated_tokens.append(next_token)
                
                # Update input_ids
                input_ids = mx.concatenate([input_ids, mx.array([next_token])])
                
                logger.info(f"Generated token: {next_token} ('{self.tokenizer.decode([next_token])}')")
        
        # Decode generated tokens
        if self.rank == 0:
            generated_text = self.tokenizer.decode(generated_tokens)
            full_text = self.tokenizer.decode(input_ids.tolist())
            logger.info(f"Generated text: '{generated_text}'")
            return full_text
        
        return ""
    
    def monitor_gpu(self):
        """Monitor GPU usage to prove it's being used."""
        hostname = os.uname().nodename
        for i in range(3):
            memory = mx.get_active_memory() / (1024**3)
            logger.info(f"GPU memory on {hostname}: {memory:.2f} GB")
            
            # Do a computation to show GPU is active
            test = mx.random.uniform(shape=(1000, 1000))
            result = mx.sum(test)
            mx.eval(result)
            
            memory_after = mx.get_active_memory() / (1024**3)
            logger.info(f"After test computation on {hostname}: {memory_after:.2f} GB")
            time.sleep(1)


def main():
    """Main entry point."""
    # Create pipeline model
    pipeline = RealPipelineModel()
    
    # Monitor GPU to show it's active
    pipeline.monitor_gpu()
    
    # Test generation (only rank 0 initiates)
    if pipeline.rank == 0:
        logger.info("=" * 50)
        logger.info("Starting generation test...")
        logger.info("=" * 50)
        
        result = pipeline.generate("What is 2+2? The answer is", max_tokens=10)
        
        logger.info("=" * 50)
        logger.info(f"Final result: {result}")
        logger.info("=" * 50)
    else:
        # Rank 1 participates in generation
        logger.info("Rank 1 ready to process layers 14-27")
        pipeline.generate("", max_tokens=10)  # Participate in pipeline


if __name__ == "__main__":
    main()