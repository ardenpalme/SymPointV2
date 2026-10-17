"""Validate converted ArchCAD ``*_s2.json`` files and report label statistics.

Every file is loaded through ``SVGDataset.load`` (the training loader).
Optionally renders a few slices next to the dataset PNGs for a visual check.

Usage:
    python archcad/check_data.py data/archcad --nproc 6 \
        --render 3 --png-dir /home/ubuntu/dataset/ArchCAD/data/png --render-out <dir>
"""

import argparse
import json
import os
import os.path as osp
import sys
from collections import Counter
from glob import glob
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from svgnet.data.archcad_categories import ARCHCAD_BG_ID, ARCHCAD_CATEGORIES
from svgnet.data.svg3 import SVGDataset

IS_THING = np.array([c["isthing"] for c in ARCHCAD_CATEGORIES], dtype=bool)


def check(json_file):
    try:
        with open(json_file) as f:
            data = json.load(f)
        n = len(data["args"])
        keys = (
            "commands",
            "lengths",
            "semanticIds",
            "instanceIds",
            "layerIds",
            "widths",
        )
        assert all(len(data[k]) == n for k in keys), "per-primitive key lengths differ"
        assert all(len(a) == 8 for a in data["args"]), "args must have 8 floats"
        assert min(data["widths"]) > 0, "widths must be > 0"
        coord, feat, label, _, _ = SVGDataset.load(json_file, 0, bg_id=ARCHCAD_BG_ID)
        sem, ins = label[:n, 0], label[:n, 1]
        assert sem.min() >= 0 and sem.max() <= ARCHCAD_BG_ID, "semantic id out of range"
        assert (ins[~IS_THING[sem]] == -1).all(), "stuff/bg primitive with an instance"
        assert np.isfinite(coord).all() and np.isfinite(feat).all(), (
            "non-finite features"
        )
        thing = IS_THING[sem]
        n_inst = len({(s, i) for s, i in zip(sem[thing], ins[thing]) if i >= 0})
        inst_classes = Counter(
            s for s, _ in {(s, i) for s, i in zip(sem[thing], ins[thing]) if i >= 0}
        )
        return (
            json_file,
            None,
            n,
            Counter(sem.tolist()),
            inst_classes,
            n_inst,
            int(((ins == -1) & thing).sum()),
        )
    except Exception as e:  # noqa: BLE001 - report every bad file, keep going
        return json_file, repr(e), 0, Counter(), Counter(), 0, 0


def render(json_file, png_dir, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    with open(json_file) as f:
        data = json.load(f)
    stem = osp.basename(json_file)[: -len("_s2.json")]
    pts = np.array(data["args"]).reshape(-1, 4, 2)
    colors = (
        np.array([ARCHCAD_CATEGORIES[s]["color"] for s in data["semanticIds"]]) / 255.0
    )
    colors[np.array(data["semanticIds"]) == ARCHCAD_BG_ID] = 0.75

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].add_collection(LineCollection(pts, colors=colors, linewidths=0.6))
    axes[0].set_xlim(0, data["width"])
    axes[0].set_ylim(data["height"], 0)
    axes[0].set_aspect("equal")
    axes[0].set_title("converted (coloured by class, bg grey)")
    png = osp.join(png_dir, stem + ".png")
    if osp.exists(png):
        axes[1].imshow(plt.imread(png))
    axes[1].set_title("dataset png")
    for ax in axes:
        ax.axis("off")
    present = sorted(set(data["semanticIds"]) - {ARCHCAD_BG_ID})
    handles = [
        plt.Line2D([], [], color=np.array(ARCHCAD_CATEGORIES[s]["color"]) / 255.0)
        for s in present
    ]
    axes[0].legend(
        handles,
        [ARCHCAD_CATEGORIES[s]["name"] for s in present],
        fontsize=7,
        loc="lower left",
    )
    out = osp.join(out_dir, stem + ".png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="converted data root with train/val/test subdirs")
    parser.add_argument("--nproc", type=int, default=os.cpu_count())
    parser.add_argument("--render", type=int, default=0, help="render N val slices")
    parser.add_argument("--png-dir", default="/home/ubuntu/dataset/ArchCAD/data/png")
    parser.add_argument("--render-out", default="data/archcad/render")
    args = parser.parse_args()

    names = [c["name"] for c in ARCHCAD_CATEGORIES]
    n_errors = 0
    for split in ("train", "val", "test"):
        files = sorted(glob(osp.join(args.root, split, "*_s2.json")))
        prims, insts = Counter(), Counter()
        n_prims, max_prims, max_inst, no_inst = 0, 0, 0, 0
        with Pool(args.nproc) as pool:
            for f, err, n, sem, inst_cls, n_inst, n_no_inst in pool.imap_unordered(
                check, files, chunksize=64
            ):
                if err:
                    n_errors += 1
                    print(f"ERROR {f}: {err}")
                    continue
                prims.update(sem)
                insts.update(inst_cls)
                n_prims += n
                max_prims = max(max_prims, n)
                max_inst = max(max_inst, n_inst)
                no_inst += n_no_inst
        print(
            f"\n== {split}: {len(files)} files, {n_prims} primitives (max {max_prims}/file), "
            f"max {max_inst} instances/file, {no_inst} thing primitives without instance"
        )
        print(f"  {'id':>2} {'class':<18} {'primitives':>11} {'instances':>10}")
        for sem in range(len(names)):
            print(f"  {sem:2d} {names[sem]:<18} {prims[sem]:11d} {insts[sem]:10d}")

    if args.render:
        os.makedirs(args.render_out, exist_ok=True)
        for f in sorted(glob(osp.join(args.root, "val", "*_s2.json")))[: args.render]:
            print("rendered", render(f, args.png_dir, args.render_out))

    print(f"\n{n_errors} files failed")
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
