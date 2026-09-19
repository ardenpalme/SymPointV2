#!/usr/bin/env bash

export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
GPUS=1
workdir=spv2-rep
OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/test.py \
	 $workdir/svg_pointT.yaml  $workdir/best.pth --dist
