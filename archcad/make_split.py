"""Deterministic train/val/test split of the ArchCAD slices.

The public release names slices by UUID only (no source-drawing id), so the
split is per slice.

Usage:
    python archcad/make_split.py /home/ubuntu/dataset/ArchCAD/data/svg \
        --out data/archcad/split.json --seed 0 --ratios 0.9 0.05 0.05
"""

import argparse
import json
import os
import os.path as osp
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="directory of ArchCAD *.svg files")
    parser.add_argument("--out", default="data/archcad/split.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.9, 0.05, 0.05])
    args = parser.parse_args()

    stems = sorted(f[:-4] for f in os.listdir(args.src) if f.endswith(".svg"))
    random.Random(args.seed).shuffle(stems)
    total = sum(args.ratios)
    n_train = round(len(stems) * args.ratios[0] / total)
    n_val = round(len(stems) * args.ratios[1] / total)
    split = {
        "train": sorted(stems[:n_train]),
        "val": sorted(stems[n_train : n_train + n_val]),
        "test": sorted(stems[n_train + n_val :]),
    }

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f)
    print(", ".join(f"{k}: {len(v)}" for k, v in split.items()), "->", args.out)


if __name__ == "__main__":
    main()
