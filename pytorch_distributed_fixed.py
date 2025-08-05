#!/usr/bin/env python3
"""
Fixed PyTorch Distributed Server for Thunderbolt Network
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket
import datetime
import logging
import time
import threading
import queue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Rank %(rank)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class RankLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {**kwargs, 'extra': {'rank': self.extra.get('rank', '?')}}

logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': '?'})

class DistributedModelShard:
    """Manages a shard of the model on one device"""
    
    def __init__(self, model_name: str, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.model_name = model_name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Update logger
        global logger
        logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': rank})
        
        logger.info(f"Initializing model shard on {self.device}")
        
        # Load tokenizer (all ranks need this)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and shard model
        self._load_and_shard_model()
        
    def _load_and_shard_model(self):
        """Load model and keep only assigned layers"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Check if it's an MLX model
        if "mlx-community" in self.model_name:
            # Use our MLX adapter
            from src.utils.mlx_pytorch_adapter import load_mlx_model_for_pytorch
            self.model, _ = load_mlx_model_for_pytorch(self.model_name, self.device)
        else:
            # Load regular PyTorch model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
                device_map={"": self.device}
            )
        
        # Get model architecture info
        if hasattr(self.model, 'base_model'):  # MLX adapter model
            self.model = self.model.base_model
        
        if hasattr(self.model, 'model'):  # Qwen models
            transformer = self.model.model
            self.layers = transformer.layers
            self.embed_tokens = transformer.embed_tokens
            self.norm = transformer.norm
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, 'transformer'):  # GPT models
            transformer = self.model.transformer
            self.layers = transformer.h
            self.embed_tokens = transformer.wte
            self.norm = transformer.ln_f
            self.lm_head = self.model.lm_head
        else:
            raise ValueError("Unsupported model architecture")
        
        total_layers = len(self.layers)
        layers_per_rank = total_layers // self.world_size
        remainder = total_layers % self.world_size
        
        # Calculate layer assignment
        if self.rank < remainder:
            start_layer = self.rank * (layers_per_rank + 1)
            end_layer = start_layer + layers_per_rank + 1
        else:
            start_layer = self.rank * layers_per_rank + remainder
            end_layer = start_layer + layers_per_rank
        
        logger.info(f"Total layers: {total_layers}, assigned layers: {start_layer}-{end_layer-1}")
        
        # Keep only assigned layers
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.assigned_layers = nn.ModuleList(self.layers[start_layer:end_layer])
        
        # Clear full model layers to save memory
        self.layers = None
        self.model = None
        
        logger.info(f"Model shard initialized with {len(self.assigned_layers)} layers")
        
    def forward_shard(self, hidden_states: torch.Tensor, attention_mask=None):
        """Forward pass through this shard's layers"""
        for layer in self.assigned_layers:
            if hasattr(layer, 'forward'):
                if attention_mask is not None:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
                else:
                    hidden_states = layer(hidden_states)[0]
            else:
                hidden_states = layer(hidden_states)
        return hidden_states

def setup_distributed():
    """Initialize distributed environment with proper network configuration"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Update logger
    global logger
    logger = RankLoggerAdapter(logging.getLogger(__name__), {'rank': rank})
    
    if world_size == 1:
        logger.info("Running in single-node mode")
        return rank, world_size
    
    # Master address configuration
    master_addr = os.environ.get('MASTER_ADDR', '192.168.5.1')
    master_port = os.environ.get('MASTER_PORT', '29501')
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # CRITICAL: For macOS with multiple interfaces, we need to help Gloo
    # find the right interface by using the IP address
    if rank == 0:
        # Master should bind to the specific Thunderbolt IP
        os.environ['GLOO_SOCKET_IFNAME'] = ''  # Let it auto-detect
        init_method = f'tcp://0.0.0.0:{master_port}'
    else:
        # Workers connect to master
        os.environ['GLOO_SOCKET_IFNAME'] = ''  # Let it auto-detect
        init_method = f'tcp://{master_addr}:{master_port}'
        # Give master time to start
        logger.info("Worker waiting 3s for master...")
        time.sleep(3)
    
    logger.info(f"Initializing distributed: rank={rank}, world_size={world_size}")
    logger.info(f"Master: {master_addr}:{master_port}")
    logger.info(f"Init method: {init_method}")
    
    try:
        dist.init_process_group(
            backend='gloo',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )
        logger.info("✓ Distributed initialization successful")
        
        # Test communication
        if rank == 0:
            test_tensor = torch.tensor([42.0])
            dist.send(test_tensor, dst=1)
            logger.info("✓ Test send successful")
        elif rank == 1:
            test_tensor = torch.zeros(1)
            dist.recv(test_tensor, src=0)
            logger.info(f"✓ Test receive successful: {test_tensor.item()}")
            
    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}")
        raise
    
    return rank, world_size

class SimpleInferenceEngine:
    """Simplified inference engine for testing"""
    
    def __init__(self, model_name: str, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.model_shard = DistributedModelShard(model_name, rank, world_size)
        
    def simple_forward_test(self):
        """Test simple forward pass"""
        if self.rank == 0:
            logger.info("Testing simple forward pass...")
            
            # Create dummy input
            batch_size = 1
            seq_len = 10
            hidden_size = self.model_shard.embed_tokens.embedding_dim
            
            # Generate embeddings
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            hidden_states = self.model_shard.embed_tokens(input_ids.to(self.model_shard.device))
            logger.info(f"Embeddings shape: {hidden_states.shape}")
            
            # Process through rank 0's layers
            hidden_states = self.model_shard.forward_shard(hidden_states)
            logger.info(f"After rank 0 layers: {hidden_states.shape}")
            
            if self.world_size > 1:
                # Send to rank 1
                hidden_states_cpu = hidden_states.cpu()
                dist.send(hidden_states_cpu, dst=1)
                logger.info("Sent to rank 1")
                
                # Wait for final result
                logits_shape = [batch_size, seq_len, self.model_shard.tokenizer.vocab_size]
                logits_cpu = torch.zeros(logits_shape)
                dist.recv(logits_cpu, src=1)
                logger.info(f"Received logits: {logits_cpu.shape}")
            else:
                # Single node
                hidden_states = self.model_shard.norm(hidden_states)
                logits = self.model_shard.lm_head(hidden_states)
                logger.info(f"Single node logits: {logits.shape}")
                
        elif self.rank == 1:
            logger.info("Worker waiting for data...")
            
            # Receive hidden states
            # We don't know the exact shape, so receive a probe first
            hidden_states_cpu = torch.zeros(1, 10, 2048)  # Dummy shape
            dist.recv(hidden_states_cpu, src=0)
            logger.info(f"Received hidden states: {hidden_states_cpu.shape}")
            
            hidden_states = hidden_states_cpu.to(self.model_shard.device)
            
            # Process through rank 1's layers
            hidden_states = self.model_shard.forward_shard(hidden_states)
            logger.info(f"After rank 1 layers: {hidden_states.shape}")
            
            # Apply final layers
            hidden_states = self.model_shard.norm(hidden_states)
            logits = self.model_shard.lm_head(hidden_states)
            logger.info(f"Generated logits: {logits.shape}")
            
            # Send back to rank 0
            logits_cpu = logits.cpu()
            dist.send(logits_cpu, dst=0)
            logger.info("Sent logits to rank 0")

def main():
    """Main entry point"""
    try:
        # Setup distributed
        rank, world_size = setup_distributed()
        
        # Model configuration
        model_name = os.environ.get('MODEL_NAME', 'mlx-community/Qwen3-1.7B-8bit')
        
        # Initialize engine
        logger.info(f"Creating inference engine for {model_name}")
        engine = SimpleInferenceEngine(model_name, rank, world_size)
        
        # Run simple test
        engine.simple_forward_test()
        
        # Synchronize before cleanup
        if world_size > 1:
            dist.barrier()
            logger.info("✓ Barrier passed")
        
        # Cleanup
        if world_size > 1:
            dist.destroy_process_group()
        
        logger.info("✓ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()