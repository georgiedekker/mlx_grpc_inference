#!/usr/bin/env python3
"""Simple PyTorch distributed test with proper macOS configuration"""
import os
import torch
import torch.distributed as dist
import socket
import datetime

def test_distributed():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"\n{'='*60}")
    print(f"Simple PyTorch Distributed Test")
    print(f"{'='*60}")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Rank: {rank}")
    print(f"World Size: {world_size}")
    
    # IMPORTANT: Use different addresses for binding vs connecting
    if rank == 0:
        # Master binds to all interfaces
        os.environ['MASTER_ADDR'] = '0.0.0.0'  # Bind to all
        init_method = 'tcp://0.0.0.0:12355'
    else:
        # Workers connect to specific IP
        os.environ['MASTER_ADDR'] = '192.168.5.1'
        init_method = 'tcp://192.168.5.1:12355'
    
    os.environ['MASTER_PORT'] = '12355'
    
    # Set interface explicitly
    gloo_interface = os.environ.get('GLOO_SOCKET_IFNAME', 'bridge0')
    print(f"GLOO_SOCKET_IFNAME: {gloo_interface}")
    print(f"Init method: {init_method}")
    print(f"{'='*60}\n")
    
    try:
        print(f"Rank {rank}: Initializing process group...")
        
        dist.init_process_group(
            backend='gloo',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )
        
        print(f"Rank {rank}: ✅ Initialized successfully!")
        
        # Simple all-reduce test
        tensor = torch.ones(1) * (rank + 1)
        print(f"Rank {rank}: Local value = {tensor.item()}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1))
        print(f"Rank {rank}: After all-reduce = {tensor.item()} (expected {expected})")
        
        # Barrier to sync
        dist.barrier()
        print(f"Rank {rank}: ✅ Test completed!")
        
        # Cleanup
        dist.destroy_process_group()
        print(f"Rank {rank}: Process group destroyed")
        
    except Exception as e:
        print(f"Rank {rank}: ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(test_distributed())