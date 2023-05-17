#!/bin/bash -x
source ../.env.sh
#export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0
export IREE_SPMD_NPROCS=2
export IREE_SPMD_PROCID=0
# Same as PROCID since we only run a single device for each process
export IREE_SPMD_RANK_OFFSET=${IREE_SPMD_PROCID}
#export NCCL_COMM_ID=127.0.0.1:54321
export JAX_PLATFORMS=iree_cuda
rm -f /tmp/pjrt/default
python -- test_distributed_pmap.py "127.0.0.1:12345" 0 2 0
rm -f /tmp/pjrt/default
