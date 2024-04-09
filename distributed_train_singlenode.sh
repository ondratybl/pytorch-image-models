#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 train.py "$@"