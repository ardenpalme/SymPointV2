#!/usr/bin/env bash
# Fine-tune SymPointV2 on the private sheets (data/private, see private/). Run from the repo root.
#   NGPU=1 EXP=private CONFIG=configs/svg/svg_pointT_private.yaml bash tools/train_private.sh [extra train.py args]
EXP=${EXP:-private} CONFIG=${CONFIG:-configs/svg/svg_pointT_private.yaml} exec bash "$(dirname "$0")/train_archcad.sh" "$@"
