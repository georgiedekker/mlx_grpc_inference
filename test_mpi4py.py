#!/usr/bin/env python3
"""
Test mpi4py communication
"""
from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

print(f"MPI4PY: Rank {rank}/{size} on {hostname}")

# Test communication
if rank == 0:
    data = {'msg': 'Hello from rank 0'}
    comm.send(data, dest=1)
    print(f"Rank 0: Sent message")
elif rank == 1:
    data = comm.recv(source=0)
    print(f"Rank 1: Received: {data}")

# Barrier to sync
comm.Barrier()
print(f"Rank {rank}: Done")