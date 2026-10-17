"""Convert ArchCAD SVG slices to SymPointV2 ``*_s2.json`` files.

Output keys and geometry follow ``parse_svg_v5.parse_svg``. ArchCAD specifics:
- ``semantic`` is the ArchCAD id (0-29, 100 = Others); Others/missing -> bg (30).
- ``instance`` strings become per-file ints; stuff and bg get -1.
- No ``<g>`` layers, so every primitive has layerId 0.
- ``transform="rotate(deg,cx,cy)"`` (used on ellipses) is applied to the sample points.

Usage:
    python archcad/convert_archcad.py /home/ubuntu/dataset/ArchCAD/data/svg \
        --split-file data/archcad/split.json --out data/archcad --nproc 6
"""

import argparse
import json
import math
import os
import os.path as osp
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from multiprocessing import Pool

from svgpathtools import parse_path

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from svgnet.data.archcad_categories import (
    ARCHCAD_BG_ID,
    ARCHCAD_CATEGORIES,
    ARCHCAD_OTHERS_ID,
)

COMMANDS = ["Line", "Arc", "circle", "ellipse"]
MIN_LENGTH = 1e-6
ROTATE_RE = re.compile(
    r"^\s*rotate\(\s*([^,\s]+)[,\s]+([^,\s]+)[,\s]+([^,\s)]+)\s*\)\s*$"
)
IS_THING = [c["isthing"] for c in ARCHCAD_CATEGORIES]


def semantic_id(attrib):
    if "semantic" not in attrib:
        return ARCHCAD_BG_ID
    sem = int(attrib["semantic"])
    if sem == ARCHCAD_OTHERS_ID:
        return ARCHCAD_BG_ID
    if not 0 <= sem < ARCHCAD_BG_ID:
        raise ValueError(f"unknown semantic {sem}")
    return sem


def stroke_rgb(attrib):
    digits = re.findall(r"\d+", attrib.get("stroke", ""))
    return [int(d) for d in digits[:3]] if len(digits) >= 3 else [0, 0, 0]


def stroke_width(attrib):
    width = float(attrib.get("stroke-width", 1.0))
    return width if width > 0 else 1.0


def rotation(attrib):
    """Point transform for an optional rotate(deg,cx,cy) attribute."""
    transform = attrib.get("transform")
    if not transform:
        return lambda x, y: (x, y)
    m = ROTATE_RE.match(transform)
    if not m:
        raise ValueError(f"unsupported transform {transform}")
    deg, ox, oy = (float(v) for v in m.groups())
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return lambda x, y: (
        ox + c * (x - ox) - s * (y - oy),
        oy + s * (x - ox) + c * (y - oy),
    )


def primitives(elem, tag):
    """Yield (command, length, 8 sample coords) for one SVG element."""
    rot = rotation(elem.attrib)
    if tag == "path":
        try:
            path = parse_path(elem.attrib["d"])
        except AssertionError:
            # svgpathtools rejects zero-length arcs (start == end); they're degenerate anyway
            return
        for seg in path:
            name = seg.__class__.__name__
            if name not in ("Line", "Arc"):
                raise ValueError(f"unsupported segment {name}")
            arg = []
            for t in (0, 1 / 3, 2 / 3, 1.0):
                p = seg.point(t)
                arg.extend(rot(p.real, p.imag))
            yield COMMANDS.index(name), seg.length(), arg
        return
    cx, cy = float(elem.attrib["cx"]), float(elem.attrib["cy"])
    if tag == "circle":
        rx = ry = float(elem.attrib["r"])
        length = 2 * math.pi * rx
    else:
        rx, ry = float(elem.attrib["rx"]), float(elem.attrib["ry"])
        a, b = max(rx, ry), min(rx, ry)
        length = 2 * math.pi * b + 4 * (a - b)
    arg = []
    for theta in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        arg.extend(rot(cx + rx * math.cos(theta), cy + ry * math.sin(theta)))
    yield COMMANDS.index(tag), length, arg


def parse_svg(svg_file):
    root = ET.parse(svg_file).getroot()
    _, _, width, height = [float(x) for x in root.attrib["viewBox"].split()]

    commands, args, lengths, widths, strokes = [], [], [], [], []
    semanticIds, instanceIds, layerIds = [], [], []
    inst_infos = defaultdict(list)
    inst_map = {}
    no_instance = 0
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag not in ("path", "circle", "ellipse"):
            continue
        sem = semantic_id(elem.attrib)
        inst = -1
        if IS_THING[sem]:
            key = elem.attrib.get("instance")
            if key is None:
                no_instance += 1
            else:
                inst = inst_map.setdefault(key, len(inst_map))
        rgb, sw = stroke_rgb(elem.attrib), stroke_width(elem.attrib)
        for command, length, arg in primitives(elem, tag):
            if length < MIN_LENGTH:
                continue
            commands.append(command)
            args.append(arg)
            lengths.append(length)
            semanticIds.append(sem)
            instanceIds.append(inst)
            layerIds.append(0)
            strokes.append(rgb)
            widths.append(sw)
            inst_infos[(inst, sem)].extend(arg)

    obj_cts, obj_boxes = [], []
    for (inst_id, sem_id), coords in inst_infos.items():
        if inst_id < 0:
            continue
        xs, ys = coords[0::2], coords[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        obj_cts.append([(x1 + x2) / 2, (y1 + y2) / 2, 0, inst_id])
        obj_boxes.append([x1, y1, x2, y2, sem_id])

    json_dict = {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semanticIds,
        "instanceIds": instanceIds,
        "width": width,
        "height": height,
        "obj_cts": obj_cts,
        "boxes": obj_boxes,
        "rgb": strokes,
        "layerIds": layerIds,
        "widths": widths,
    }
    return json_dict, no_instance


def process(job):
    svg_file, out_json = job
    try:
        json_dict, no_instance = parse_svg(svg_file)
        if not json_dict["args"]:
            raise ValueError("no primitives")
        with open(out_json, "w") as f:
            json.dump(json_dict, f)
        return (
            svg_file,
            None,
            len(json_dict["args"]),
            Counter(json_dict["semanticIds"]),
            no_instance,
        )
    except Exception:  # noqa: BLE001 - log the failing file, keep converting
        return svg_file, traceback.format_exc(limit=2), 0, Counter(), 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="directory of ArchCAD *.svg files")
    parser.add_argument("--split-file", default="data/archcad/split.json")
    parser.add_argument("--out", default="data/archcad")
    parser.add_argument("--nproc", type=int, default=os.cpu_count())
    parser.add_argument(
        "--limit", type=int, default=0, help="convert only the first N stems per split"
    )
    args = parser.parse_args()

    with open(args.split_file) as f:
        split = json.load(f)
    jobs = []
    for name, stems in split.items():
        os.makedirs(osp.join(args.out, name), exist_ok=True)
        for stem in stems[: args.limit or None]:
            jobs.append(
                (
                    osp.join(args.src, stem + ".svg"),
                    osp.join(args.out, name, stem + "_s2.json"),
                )
            )

    errors, n_prims, sem_counts, n_no_inst = [], 0, Counter(), 0
    with Pool(args.nproc) as pool:
        for i, (svg_file, err, n, counts, no_inst) in enumerate(
            pool.imap_unordered(process, jobs, chunksize=32)
        ):
            if err:
                errors.append(f"{svg_file}\n{err}")
            n_prims += n
            sem_counts.update(counts)
            n_no_inst += no_inst
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)

    with open(osp.join(args.out, "convert_errors.txt"), "w") as f:
        f.write("\n".join(errors))
    print(
        f"converted {len(jobs) - len(errors)}/{len(jobs)} files, {len(errors)} errors, {n_prims} primitives"
    )
    print(f"thing-class elements without an instance attribute: {n_no_inst}")
    for sem in sorted(sem_counts):
        print(f"  {sem:2d} {ARCHCAD_CATEGORIES[sem]['name']:<18} {sem_counts[sem]}")


if __name__ == "__main__":
    main()
