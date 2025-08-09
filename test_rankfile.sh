#!/bin/bash

echo "Testing MPI with rankfile mapping..."
echo ""
echo "Rankfile contents:"
cat rankfile
echo ""
echo "Hosts.json contents:"
cat hosts.json
echo ""
echo "Testing MPI connectivity with rankfile..."
echo ""

# Simple MPI test with rankfile
mpirun --hostfile mpi_hostfile \
       --map-by rankfile:file=rankfile \
       -n 2 \
       hostname

echo ""
echo "If hostnames show mini1.local and mini2.local,"
echo "then rankfile mapping is working correctly."