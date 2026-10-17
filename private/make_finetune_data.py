"""Build the private fine-tuning set from labelled pre-processing-pipeline windows.

Input per sheet (``out/<pdf stem>/p<NNN>``, written by ``pre-processing-pipeline/main.py``):
- ``windows/<id>_s2.json``: 140x140 tiles, already labelled from the CVAT boxes (FloorPlanCAD ids, bg 35).
- ``windows.json``: id -> {bbox (sheet pt), row, col, prims (sheet primitive index per tile primitive)}.
- ``sem_ins_split_val.npy``: ``tools/inference.py`` output of the pretrained model on those windows.

Labels:
- Box labels are kept as they are.
- Classes the annotators covered (doors, windows, stairs, elevator, escalator) come only from boxes.
- Other classes (furniture, sanitary, walls, ...) are pseudo-labelled from confident predictions, on
  primitives the boxes left as background. Row chairs are not: the model gives that class to desk chairs.

Split: windows in ``--val-rows`` go to val, the rest to train. Train windows lose every primitive that lies
in a val window (and every instance touching one), so no sheet primitive is in both splits. ``obj_cts`` and
``boxes`` are copied unchanged; the loader does not read them.

Usage:
    python private/make_finetune_data.py \
        --sheet "/home/ubuntu/out/A-1-03-80 - LEVEL 03 - GENERAL ARRANGEMENT PLAN/p000" \
        --out data/private --val-rows 5,6 --render
"""

import argparse
import json
import os
import os.path as osp
import re
import sys
from collections import Counter
from glob import glob

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from svgnet.data.svg3 import SVG_CATEGORIES, SVGDataset

BG_ID = len(SVG_CATEGORIES) - 1
IS_THING = np.array([c["isthing"] for c in SVG_CATEGORIES], dtype=bool)
# Annotated in CVAT (or would have been if present): doors/windows 0-9, stairs/elevator/escalator 27-29.
COVERED = set(range(10)) | {27, 28, 29}
# Never pseudo-labelled: the pretrained model calls office desk chairs "row chairs".
NO_PSEUDO = COVERED | {30, BG_ID}
PSEUDO_SCORE = 0.7
PSEUDO_INST_BASE = 100000
MIN_PRIMS = 50
# pre-processing-pipeline/windows.py: a 441 pt window maps to a 140-unit tile.
WINDOW_PT = 441.0
TILE = 140.0
PER_PRIM_KEYS = (
    "commands",
    "args",
    "lengths",
    "semanticIds",
    "instanceIds",
    "rgb",
    "layerIds",
    "widths",
)


def load_predictions(npy):
    """Window id -> list of predicted instances {masks, labels, scores}."""
    preds = {}
    for d in np.load(npy, allow_pickle=True):
        wid = osp.basename(d["filepath"]).split("_")[0]
        preds[wid] = d["ins"]
    return preds


def pseudo_label(sem, ins, preds):
    """Paints confident non-covered predictions onto box-background primitives, in place.

    Higher scores are painted last, so they win. Returns the number of pseudo instances used.
    """
    n = len(sem)
    free = sem == BG_ID
    used = 0
    for k, p in sorted(enumerate(preds), key=lambda kp: kp[1]["scores"]):
        label = int(p["labels"])
        if label in NO_PSEUDO or p["scores"] < PSEUDO_SCORE:
            continue
        m = np.asarray(p["masks"][:n], dtype=bool) & free
        if not m.any():
            continue
        sem[m] = label
        ins[m] = PSEUDO_INST_BASE + k if IS_THING[label] else -1
        used += 1
    return used


def sheet_points(args, bbox):
    """(N, 4, 2) sample points of the tile primitives in sheet pt."""
    pts = np.asarray(args, dtype=float).reshape(-1, 4, 2)
    return pts * (WINDOW_PT / TILE) + np.array(bbox[:2])


def in_any_bbox(pts, bboxes):
    """(N,) True where any of a primitive's points lies in any of the bboxes."""
    hit = np.zeros(len(pts), dtype=bool)
    for x0, y0, x1, y1 in bboxes:
        inside = (
            (pts[..., 0] >= x0)
            & (pts[..., 0] <= x1)
            & (pts[..., 1] >= y0)
            & (pts[..., 1] <= y1)
        )
        hit |= inside.any(axis=1)
    return hit


def build_sheet(sheet_dir, val_rows):
    """Labelled windows of one sheet as (split, window id, data, sheet prim indices)."""
    with open(osp.join(sheet_dir, "windows.json")) as f:
        meta = json.load(f)
    preds = load_predictions(osp.join(sheet_dir, "sem_ins_split_val.npy"))
    val_ids = {wid for wid, m in meta.items() if m["row"] in val_rows}
    val_bboxes = [meta[wid]["bbox"] for wid in val_ids]
    val_prims = {p for wid in val_ids for p in meta[wid]["prims"]}

    out, n_pseudo = [], 0
    for wid in sorted(meta, key=int):
        with open(osp.join(sheet_dir, "windows", f"{wid}_s2.json")) as f:
            data = json.load(f)
        n = len(data["args"])
        if n < MIN_PRIMS:
            continue
        sem = np.array(data["semanticIds"], dtype=np.int64)
        ins = np.array(data["instanceIds"], dtype=np.int64)
        n_pseudo += pseudo_label(sem, ins, preds.get(wid, []))
        prims = np.array(meta[wid]["prims"])
        keep = np.ones(n, dtype=bool)
        if wid not in val_ids and val_ids:
            drop = in_any_bbox(
                sheet_points(data["args"], meta[wid]["bbox"]), val_bboxes
            )
            drop |= np.isin(prims, list(val_prims))
            cut = {i for i in ins[drop & (ins >= 0)].tolist()}
            drop |= np.isin(ins, list(cut))
            keep = ~drop
        data["semanticIds"] = sem.tolist()
        data["instanceIds"] = ins.tolist()
        for key in PER_PRIM_KEYS:
            data[key] = [v for v, k in zip(data[key], keep) if k]
        if len(data["args"]) < MIN_PRIMS:
            continue
        split = "val" if wid in val_ids else "train"
        out.append((split, wid, data, prims[keep]))
    return out, n_pseudo


def check(json_file):
    """The ``archcad/check_data.py`` assertions, for FloorPlanCAD ids. Returns (n, sem counts, instances)."""
    with open(json_file) as f:
        data = json.load(f)
    n = len(data["args"])
    assert all(len(data[k]) == n for k in PER_PRIM_KEYS), (
        "per-primitive key lengths differ"
    )
    assert all(len(a) == 8 for a in data["args"]), "args must have 8 floats"
    assert min(data["widths"]) > 0, "widths must be > 0"
    coord, feat, label, _, _ = SVGDataset.load(json_file, 0, bg_id=BG_ID)
    sem, ins = label[:n, 0], label[:n, 1]
    assert sem.min() >= 0 and sem.max() <= BG_ID, "semantic id out of range"
    assert (ins[~IS_THING[sem]] == -1).all(), "stuff/bg primitive with an instance"
    assert np.isfinite(coord).all() and np.isfinite(feat).all(), "non-finite features"
    thing = IS_THING[sem] & (ins >= 0)
    insts = {(int(s), int(i)) for s, i in zip(sem[thing], ins[thing])}
    return n, Counter(sem.tolist()), insts


def render(entries, meta_by_sheet, out_png):
    """Plots every written primitive in sheet pt, coloured by class (bg grey), val windows dashed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(30, 22))
    for sheet, split, wid, data in entries:
        bbox = meta_by_sheet[sheet][wid]["bbox"]
        pts = sheet_points(data["args"], bbox)
        sem = np.array(data["semanticIds"])
        colors = (
            np.array([c["color"] for c in SVG_CATEGORIES] + [[200, 200, 200]]) / 255
        )
        colors[BG_ID] = [0.85, 0.85, 0.85]
        order = np.argsort(sem != BG_ID)  # bg first, labels on top
        ax.add_collection(
            LineCollection(pts[order], colors=colors[sem[order]], linewidths=0.3)
        )
        if split == "val":
            x0, y0, x1, y1 = bbox
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=0.8, ec="k"
                )
            )
    ax.autoscale()
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def slug(sheet_dir):
    sheet_dir = osp.normpath(sheet_dir)
    name = f"{osp.basename(osp.dirname(sheet_dir))}_{osp.basename(sheet_dir)}"
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sheet", action="append", required=True, help="out/<stem>/p<NNN>"
    )
    parser.add_argument("--out", default="data/private")
    parser.add_argument(
        "--val-rows", default="5,6", help="window rows for val ('' = none)"
    )
    parser.add_argument(
        "--render", action="store_true", help="write <out>/render/<sheet>.png"
    )
    args = parser.parse_args()
    val_rows = {int(r) for r in args.val_rows.split(",") if r.strip()}

    for split in ("train", "val"):
        os.makedirs(osp.join(args.out, split), exist_ok=True)
        for f in glob(osp.join(args.out, split, "*_s2.json")):
            os.remove(f)

    split_info, n_errors = {}, 0
    for sheet_dir in args.sheet:
        name = slug(sheet_dir)
        windows, n_pseudo = build_sheet(sheet_dir, val_rows)
        used = {"train": set(), "val": set()}
        for split, wid, data, prims in windows:
            with open(osp.join(args.out, split, f"{name}_w{wid}_s2.json"), "w") as f:
                json.dump(data, f)
            used[split].update(prims.tolist())
        shared = used["train"] & used["val"]
        assert not shared, f"{name}: {len(shared)} sheet primitives in both splits"
        split_info[name] = {
            "sheet": osp.abspath(sheet_dir),
            "val_rows": sorted(val_rows),
            **{
                s: sorted((w for sp, w, _, _ in windows if sp == s), key=int)
                for s in used
            },
        }
        print(f"{name}: {len(windows)} windows, {n_pseudo} pseudo instances painted")
        if args.render:
            with open(osp.join(sheet_dir, "windows.json")) as f:
                meta = {name: json.load(f)}
            os.makedirs(osp.join(args.out, "render"), exist_ok=True)
            out_png = osp.join(args.out, "render", f"{name}.png")
            render([(name, s, w, d) for s, w, d, _ in windows], meta, out_png)
            print("rendered", out_png)

    with open(osp.join(args.out, "split.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    names = [c["name"] for c in SVG_CATEGORIES]
    for split in ("train", "val"):
        files = sorted(glob(osp.join(args.out, split, "*_s2.json")))
        prims, insts, n_prims = Counter(), set(), 0
        for f in files:
            try:
                n, sem, file_insts = check(f)
            except Exception as e:  # noqa: BLE001 - report every bad file, keep going
                n_errors += 1
                print(f"ERROR {f}: {e!r}")
                continue
            n_prims += n
            prims.update(sem)
            # Box ids are per sheet, pseudo ids per window.
            insts |= {
                (
                    osp.basename(f).rsplit("_w", 1)[0] if i < PSEUDO_INST_BASE else f,
                    s,
                    i,
                )
                for s, i in file_insts
            }
        box = Counter(s for key, s, i in insts if i < PSEUDO_INST_BASE)
        pseudo = Counter(s for key, s, i in insts if i >= PSEUDO_INST_BASE)
        print(f"\n== {split}: {len(files)} files, {n_prims} primitives")
        print(
            f"  {'id':>2} {'class':<18} {'primitives':>11} {'box inst':>9} {'pseudo inst':>12}"
        )
        for sem in range(len(names)):
            if prims[sem]:
                print(
                    f"  {sem:2d} {names[sem]:<18} {prims[sem]:11d} {box[sem]:9d} {pseudo[sem]:12d}"
                )

    print(f"\n{n_errors} files failed")
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
