#!/usr/bin/env python3
"""
PyTorch distributed server with heterogeneous capability-based sharding.
"""
import os
import sys
import time
import json
import logging
import argparse
import socket
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.device_capability import DeviceCapability, DeviceProfiler, estimate_memory_per_layer
from utils.layer_assignment import (
    calculate_layer_distribution, 
    LayerAssignment,
    validate_assignment
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HeterogeneousModelShard(nn.Module):
    """Model shard with capability-based layer assignment."""
    
    def __init__(
        self, 
        model_name: str,
        rank: int,
        world_size: int,
        layer_assignment: Optional[LayerAssignment] = None,
        device_capabilities: Optional[Dict[str, DeviceCapability]] = None,
        sharding_strategy: str = "capability_based"
    ):
        super().__init__()
        self.model_name = model_name
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.sharding_strategy = sharding_strategy
        
        logger.info(f"Initializing shard on rank {rank} with device {self.device}")
        
        # Load model configuration
        self.config = AutoConfig.from_pretrained(model_name)
        self.num_layers = self.config.num_hidden_layers
        
        # Determine layer assignment
        if layer_assignment:
            self.layer_assignment = layer_assignment
        else:
            # Calculate assignment based on capabilities
            self._calculate_layer_assignment(device_capabilities)
        
        # Load assigned model components
        self._load_model_shard()
        
        logger.info(f"Rank {rank} initialized with {self.layer_assignment}")
    
    def _calculate_layer_assignment(self, device_capabilities: Optional[Dict[str, DeviceCapability]]):
        """Calculate layer assignment based on device capabilities."""
        if not device_capabilities or self.sharding_strategy == "equal":
            # Fall back to equal distribution
            layers_per_rank = self.num_layers // self.world_size
            remainder = self.num_layers % self.world_size
            
            start_layer = self.rank * layers_per_rank + min(self.rank, remainder)
            num_layers = layers_per_rank + (1 if self.rank < remainder else 0)
            end_layer = start_layer + num_layers - 1
            
            self.layer_assignment = LayerAssignment(
                device_id=f"rank_{self.rank}",
                rank=self.rank,
                start_layer=start_layer,
                end_layer=end_layer,
                layer_indices=list(range(start_layer, end_layer + 1)),
                has_embeddings=(self.rank == 0),
                has_lm_head=(self.rank == self.world_size - 1)
            )
        else:
            # Use capability-based assignment
            # This should be pre-calculated and passed in
            raise ValueError("Device capabilities provided but no layer assignment calculated")
    
    def _load_model_shard(self):
        """Load only the assigned layers and components."""
        logger.info(f"Loading layers {self.layer_assignment.start_layer} to {self.layer_assignment.end_layer}")
        
        # Load full model first (we'll extract what we need)
        full_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map={"": self.device}
        )
        
        # Extract embeddings if assigned
        if self.layer_assignment.has_embeddings:
            self.embeddings = full_model.get_input_embeddings()
            logger.info(f"Rank {self.rank} loaded embeddings")
        else:
            self.embeddings = None
        
        # Extract assigned transformer layers
        self.layers = nn.ModuleList()
        all_layers = full_model.model.layers if hasattr(full_model.model, 'layers') else full_model.transformer.h
        
        for idx in self.layer_assignment.layer_indices:
            self.layers.append(all_layers[idx])
        
        # Extract LM head if assigned
        if self.layer_assignment.has_lm_head:
            self.lm_head = full_model.get_output_embeddings()
            self.norm = full_model.model.norm if hasattr(full_model.model, 'norm') else full_model.transformer.ln_f
            logger.info(f"Rank {self.rank} loaded LM head and final norm")
        else:
            self.lm_head = None
            self.norm = None
        
        # Clean up full model
        del full_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Log memory usage
        if hasattr(torch.mps, 'current_allocated_memory'):
            allocated = torch.mps.current_allocated_memory() / 1024**3
            logger.info(f"Rank {self.rank} using {allocated:.2f} GB")
    
    def forward(self, hidden_states: torch.Tensor, use_cache: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass through assigned layers."""
        # Apply embeddings if this is the first shard
        if self.embeddings is not None:
            hidden_states = self.embeddings(hidden_states)
        
        # Pass through assigned layers
        for layer in self.layers:
            layer_outputs = layer(hidden_states, use_cache=use_cache)
            hidden_states = layer_outputs[0]
        
        # Apply final norm and LM head if this is the last shard
        if self.lm_head is not None:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return {"logits": logits, "hidden_states": hidden_states}
        
        return {"hidden_states": hidden_states}


class HeterogeneousDistributedModel:
    """Distributed model with heterogeneous sharding support."""
    
    def __init__(
        self,
        model_name: str,
        rank: int,
        world_size: int,
        cluster_config: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.rank = rank
        self.world_size = world_size
        self.cluster_config = cluster_config or {}
        
        # Load tokenizer on all ranks
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Profile devices and calculate layer assignments
        self._setup_heterogeneous_sharding()
        
        # Create model shard with assignment
        self.model_shard = HeterogeneousModelShard(
            model_name=model_name,
            rank=rank,
            world_size=world_size,
            layer_assignment=self.layer_assignment,
            device_capabilities=self.device_capabilities,
            sharding_strategy=self.sharding_strategy
        )
        
        logger.info(f"Rank {rank} initialized heterogeneous distributed model")
    
    def _setup_heterogeneous_sharding(self):
        """Setup device profiling and layer assignments."""
        # Get sharding configuration
        model_config = self.cluster_config.get('model', {})
        self.sharding_strategy = model_config.get('sharding_strategy', 'capability_based')
        
        if self.rank == 0:
            # Master profiles all devices and calculates assignments
            device_list = self.cluster_config.get('cluster', {}).get('devices', [])
            if not device_list:
                # Default device list
                device_list = [f"rank_{i}" for i in range(self.world_size)]
            
            # Profile devices
            self.device_capabilities = {}
            for i, device_name in enumerate(device_list[:self.world_size]):
                if i == self.rank:
                    # Profile local device
                    cap = DeviceProfiler.profile_local_device(device_id=device_name)
                else:
                    # Use predefined or default profile
                    cap = DeviceProfiler._get_predefined_profile(device_name)
                self.device_capabilities[device_name] = cap
            
            # Calculate layer assignments
            config = AutoConfig.from_pretrained(self.model_name)
            total_layers = config.num_hidden_layers
            
            # Get manual assignment if provided
            manual_assignment = model_config.get('layer_distribution')
            
            assignments = calculate_layer_distribution(
                devices=list(self.device_capabilities.values()),
                total_layers=total_layers,
                strategy=self.sharding_strategy,
                model_name=self.model_name,
                manual_assignment=manual_assignment
            )
            
            # Validate assignment
            if not validate_assignment(assignments, total_layers):
                raise ValueError("Invalid layer assignment calculated")
            
            # Broadcast assignments to all ranks
            assignment_data = {
                'capabilities': {k: v.to_dict() for k, v in self.device_capabilities.items()},
                'assignments': {k: {
                    'device_id': v.device_id,
                    'rank': v.rank,
                    'start_layer': v.start_layer,
                    'end_layer': v.end_layer,
                    'layer_indices': v.layer_indices,
                    'has_embeddings': v.has_embeddings,
                    'has_lm_head': v.has_lm_head
                } for k, v in assignments.items()}
            }
            
            # Send to all ranks
            for dst in range(1, self.world_size):
                dist.send(
                    torch.tensor(
                        [ord(c) for c in json.dumps(assignment_data)],
                        dtype=torch.uint8
                    ),
                    dst=dst
                )
            
            # Set local assignment
            device_id = device_list[0]
            self.layer_assignment = assignments[device_id]
            
        else:
            # Receive assignment from master
            max_size = 10000  # Max JSON size
            tensor = torch.zeros(max_size, dtype=torch.uint8)
            dist.recv(tensor, src=0)
            
            # Decode assignment
            json_str = ''.join(chr(c) for c in tensor if c != 0)
            assignment_data = json.loads(json_str)
            
            # Reconstruct capabilities
            self.device_capabilities = {}
            for device_id, cap_dict in assignment_data['capabilities'].items():
                self.device_capabilities[device_id] = DeviceCapability(**cap_dict)
            
            # Find our assignment
            device_list = self.cluster_config.get('cluster', {}).get('devices', [f"rank_{i}" for i in range(self.world_size)])
            device_id = device_list[self.rank]
            
            assignment_dict = assignment_data['assignments'][device_id]
            self.layer_assignment = LayerAssignment(**assignment_dict)
        
        # Log final assignment
        logger.info(f"Rank {self.rank} assignment: {self.layer_assignment}")
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate text using the distributed model."""
        # Only rank 0 handles tokenization
        if self.rank == 0:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.model_shard.device)
            
            # Broadcast input size
            input_size = torch.tensor([input_ids.shape[1]], dtype=torch.long)
            dist.broadcast(input_size, src=0)
            
            # Broadcast input_ids
            dist.broadcast(input_ids, src=0)
        else:
            # Receive input size
            input_size = torch.tensor([0], dtype=torch.long)
            dist.broadcast(input_size, src=0)
            
            # Receive input_ids
            input_ids = torch.zeros((1, input_size.item()), dtype=torch.long)
            dist.broadcast(input_ids, src=0)
            input_ids = input_ids.to(self.model_shard.device)
        
        # Generate tokens
        generated_ids = input_ids
        
        for _ in range(max_length):
            # Forward pass through pipeline
            if self.rank == 0:
                # First shard starts with input_ids
                outputs = self.model_shard(generated_ids)
                hidden_states = outputs["hidden_states"]
                
                # Send to next rank
                if self.world_size > 1:
                    dist.send(hidden_states.cpu(), dst=1)
            
            elif self.rank == self.world_size - 1:
                # Last shard receives hidden states and produces logits
                hidden_states = torch.zeros_like(generated_ids, dtype=torch.float16)
                dist.recv(hidden_states, src=self.rank - 1)
                hidden_states = hidden_states.to(self.model_shard.device)
                
                outputs = self.model_shard(hidden_states)
                logits = outputs["logits"]
                
                # Sample next token
                next_token_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Broadcast next token to all ranks
                dist.broadcast(next_token, src=self.world_size - 1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
            else:
                # Middle shards receive, process, and send
                hidden_states = torch.zeros_like(generated_ids, dtype=torch.float16)
                dist.recv(hidden_states, src=self.rank - 1)
                hidden_states = hidden_states.to(self.model_shard.device)
                
                outputs = self.model_shard(hidden_states)
                hidden_states = outputs["hidden_states"]
                
                dist.send(hidden_states.cpu(), dst=self.rank + 1)
                
                # Receive next token from last rank
                next_token = torch.zeros((1, 1), dtype=torch.long)
                dist.broadcast(next_token, src=self.world_size - 1)
                generated_ids = torch.cat([generated_ids, next_token.to(self.model_shard.device)], dim=-1)
            
            # Check for EOS token
            if generated_ids[0, -1].item() == self.tokenizer.eos_token_id:
                break
        
        # Decode on rank 0
        if self.rank == 0:
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text
        else:
            return ""


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch heterogeneous distributed inference server')
    parser.add_argument('--rank', type=int, required=True, help='Rank of this process')
    parser.add_argument('--world-size', type=int, required=True, help='Total number of processes')
    parser.add_argument('--master-addr', type=str, default='localhost', help='Master address')
    parser.add_argument('--master-port', type=str, default='29501', help='Master port')
    parser.add_argument('--model-name', type=str, default='microsoft/phi-2', help='Model to use')
    parser.add_argument('--config', type=str, help='Path to cluster configuration file')
    parser.add_argument('--bind-addr', type=str, default='0.0.0.0', help='Address to bind Gloo server')
    return parser.parse_args()


def setup_distributed(args):
    """Initialize distributed process group with proper network configuration."""
    # Set environment variables
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    
    # Configure Gloo settings for macOS
    if args.rank == 0:
        # Master binds to all interfaces
        os.environ['GLOO_SOCKET_IFNAME'] = ''
        os.environ['TP_SOCKET_IFNAME'] = ''
    
    # Initialize process group
    logger.info(f"Initializing process group: rank={args.rank}, world_size={args.world_size}")
    
    try:
        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://{args.master_addr}:{args.master_port}',
            rank=args.rank,
            world_size=args.world_size,
            timeout=dist.default_pg_timeout
        )
        logger.info(f"Process group initialized successfully on rank {args.rank}")
    except Exception as e:
        logger.error(f"Failed to initialize process group: {e}")
        raise


def main():
    args = parse_args()
    
    # Setup distributed training
    setup_distributed(args)
    
    # Load cluster configuration if provided
    cluster_config = {}
    if args.config:
        with open(args.config, 'r') as f:
            cluster_config = json.load(f)
    
    # Create distributed model
    model = HeterogeneousDistributedModel(
        model_name=args.model_name,
        rank=args.rank,
        world_size=args.world_size,
        cluster_config=cluster_config
    )
    
    # Keep process alive for API server to use
    logger.info(f"Rank {args.rank} ready for inference")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info(f"Rank {args.rank} shutting down")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()