#!/usr/bin/env bash
# Train SymPointV2 on ArchCAD (data/archcad, see archcad/). Run from the repo root.
#   NGPU=1 EXP=archcad CONFIG=configs/svg/svg_pointT_archcad.yaml bash tools/train_archcad.sh [extra train.py args]
# torchrun/--dist is needed even on one GPU: validate() calls dist.get_world_size().

export PYTHONPATH=./
PYTHON=${PYTHON:-/home/ubuntu/miniforge3/envs/spv2/bin/python}
NGPU=${NGPU:-1}
EXP=${EXP:-archcad}
CONFIG=${CONFIG:-configs/svg/svg_pointT_archcad.yaml}
SYNC_BN=""
if [ "$NGPU" -gt 1 ]; then SYNC_BN="--sync_bn"; fi

OMP_NUM_THREADS=$NGPU $PYTHON -m torch.distributed.run --nproc_per_node=$NGPU --master_port=$((RANDOM + 10000)) tools/train.py \
	--dist $CONFIG \
	--exp_name $EXP \
	$SYNC_BN "$@"
