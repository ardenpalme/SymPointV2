"""Build the measurement, tag, ALIGN marker, callout or room label fixtures: small crop PDFs of
A-1-03-10 and candidate expected groups.

Usage, from the repo root:
    python -m tests.make_fixtures --kind {measurements,tags,align,callouts,rooms} --review-dir DIR

Writes tests/fixtures/<crop>.pdf, tests/expected/<crop>/<kind>.json, and DIR/<crop>.png
with the candidate groups overlaid (one color per group, rect dashed, leader_rect dotted, room
label end dots circled). The JSON files are candidates: they are produced by get_measurements(),
get_tags(), get_align_markers(), get_callouts() or get_room_labels() itself and become expected
outputs only after a human has reviewed the PNGs.
Rerunning overwrites the crops of that kind only.

A crop keeps only the drawing items whose rect (inflated by half the stroke width) meets its
region, then sets the mediabox to the region, so crop coordinates are page coordinates
(PyMuPDF page space, pt) minus the region's top-left corner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymupdf
import shapely
from shapely import Polygon, STRtree

from data_eng.parser import JOIN_TOL, FloorPlanParser, room_label_json
from data_eng.utils import remove_items

SOURCE = Path("tmp_dataset/A-1-03-10 - LEVEL 03 - PARTITION PLAN.pdf")
FIXTURES = Path("tests/fixtures")
EXPECTED = Path("tests/expected")
CROP_MARGIN = 8.0  # pt around the groups' rects
# Crop stem -> texts whose measurement groups (found on the full page) the crop must contain.
CROPS = {
    "dim_1691": ["1691 mm"],  # text next to its dimension line
    "leader_580": ["580 mm"],  # text joined to its dimension line by a curved leader
    "chain_10030_220": [
        "10030 mm",
        "220 mm",
    ],  # two collinear dimensions sharing an extension line
    "along_wall_2950": [
        "2950 mm"
    ],  # dimension line along a wall face, wall edges at both ends
}
# Crop stem -> tags the crop must contain, each as (text, a point inside its rect on the page, pt).
# Tag texts repeat across the page, so the point picks the tag.
TAG_CROPS = {
    "tag_bent_f23": [("F23", (841.0, 330.0))],  # two-piece leader, bar on a wall next to grid lines
    "tag_straight_g03": [("G03", (1445.0, 1850.0))],  # one-piece leader
    "tag_stacked_j02": [
        ("J02", (1739.6, 1794.7)),
        ("F21", (1752.0, 1775.0)),
    ],  # J02 drawn twice (two leaders), touching F21
    "tag_touching_g03_j31": [
        ("G03", (699.6, 867.9)),
        ("J31", (687.5, 879.4)),
    ],  # parallel diamond edges 0.4 pt apart
}
# Crop stem -> ALIGN markers the crop must contain, each as (text, a point inside its rect on the page, pt).
ALIGN_CROPS = {
    "align_horizontal": [("ALIGN", (1789.0, 1830.0))],  # text below, arrowheads point up
    "align_vertical": [("ALIGN", (1367.0, 1935.0))],  # text read upward, arrowheads point right
    "align_diagonal": [("ALIGN", (730.0, 1085.0))],  # 45 deg, arrowhead clips with stair-step slivers
    "align_skewed": [("ALIGN", (2218.0, 1955.0))],  # about 22 deg
    "align_side_by_side": [
        ("ALIGN", (965.0, 1608.0)),
        ("ALIGN", (1009.0, 1608.0)),
    ],  # two markers next to each other
    "align_long": [("ALIGN", (1944.0, 1597.0))],  # 148 pt crossbar
}
# Crop stem -> callouts the crop must contain, each as (code, a point inside its rect on the page, pt).
CALLOUT_CROPS = {
    "callout_corner_one": [("A-4.03.51", (1745.0, 1239.0))],  # one black corner
    "callout_corners_three": [("A-4.03.50", (1985.0, 1580.0))],  # three black corners
    "callout_pointer_one": [("A-4.03.52", (653.0, 1222.0))],  # one pointer sticking out of the circle
    "callout_pointers_two": [("A-4.03.52", (574.0, 1216.0))],  # two pointers, wall lines through the bbox
    "callout_parts_scattered": [
        ("A-4.03.51", (1124.0, 863.0))
    ],  # parts far apart in drawing order, next to a tag and a measurement
}
# Crop stem -> room labels the crop must contain, each as (number, a point inside its rect on the page, pt).
ROOM_CROPS = {
    "room_bent_leader_03-C15": [("03-C15", (1430.0, 1000.0))],  # two-piece leader from the box bottom
    "room_straight_leader_03-A36": [("03-A36", (1551.0, 1238.0))],  # one-piece leader
    "room_no_leader_03-A20": [("03-A20", (963.0, 1400.0))],  # four name lines
    "room_no_leader_03-A55": [("03-A55", (1732.0, 1443.0))],  # one name line
    "room_leader_crosses_walls_03-C17": [("03-C17", (1532.0, 1097.0))],  # leader crosses 4 wall lines
}
COLORS = [
    (0.9, 0, 0),
    (0, 0.6, 0),
    (0, 0, 0.9),
    (0.9, 0.5, 0),
    (0.6, 0, 0.8),
    (0, 0.6, 0.6),
]


def select_measurements(groups: list[dict], texts: list[str]) -> list[dict]:
    """The one measurement group per text."""
    selected = []
    for text in texts:
        matches = [g for g in groups if g["text"] == text]
        if len(matches) != 1:
            raise ValueError(
                f"{text!r} has {len(matches)} measurement groups on the page, expected 1"
            )
        selected.append(matches[0])
    return selected


def rect_bounds(rect: list[float] | Polygon) -> pymupdf.Rect:
    """A group's rect as a pymupdf.Rect: [x0, y0, x1, y1], or the bounds of a room label's Polygon."""
    return pymupdf.Rect(rect.bounds if isinstance(rect, Polygon) else rect)


def select_tags(
    groups: list[dict], picks: list[tuple[str, tuple[float, float]]]
) -> list[dict]:
    """The one tag, ALIGN marker, callout or room label per (text, point) whose rect contains the point."""
    selected = []
    for text, point in picks:
        matches = [
            g for g in groups if g["text"] == text and pymupdf.Point(point) in rect_bounds(g["rect"])
        ]
        if len(matches) != 1:
            raise ValueError(f"{text!r} at {point} has {len(matches)} groups on the page, expected 1")
        selected.append(matches[0])
    return selected


def crop_region(groups: list[dict]) -> pymupdf.Rect:
    """Union of the groups' rects (and leader_rects and dots), plus CROP_MARGIN (page space, pt)."""
    region = pymupdf.Rect()
    for group in groups:
        region |= rect_bounds(group["rect"])
        if group.get("leader_rect") is not None:
            region |= pymupdf.Rect(group["leader_rect"])
        if group.get("dot") is not None:
            region |= pymupdf.Point(group["dot"].x, group["dot"].y)
    return region + (-CROP_MARGIN, -CROP_MARGIN, CROP_MARGIN, CROP_MARGIN)


def path_points(path: dict) -> list[tuple[float, float]]:
    """End points of a get_drawings() path's items (corners for re and qu)."""
    points = []
    for item in path["items"]:
        if item[0] in ("l", "c"):
            points += [tuple(item[1]), tuple(item[-1])]
        elif item[0] == "re":
            points += [
                tuple(item[1].tl),
                tuple(item[1].br),
                tuple(item[1].tr),
                tuple(item[1].bl),
            ]
        elif item[0] == "qu":
            points += [
                tuple(item[1].ul),
                tuple(item[1].ur),
                tuple(item[1].ll),
                tuple(item[1].lr),
            ]
        else:
            raise ValueError("unhandled drawing", item)
    return points


def make_crop(
    source: pymupdf.Document, region: pymupdf.Rect
) -> tuple[pymupdf.Document, list[int]]:
    """Page 0 of source reduced to the drawing items near region, with region as mediabox.

    Keeps the paths whose rect (inflated by half the stroke width) meets region, plus every
    path whose rect is within JOIN_TOL of an end point of those, so the heuristics see the
    same neighbors at the ends of kept lines as on the full page. Returns the crop and the
    page indices of the kept paths (crop path i is page path kept[i]).
    """
    doc = pymupdf.open()
    doc.insert_pdf(source, from_page=0, to_page=0)
    page = doc[0]
    if page.rotation != 0:
        raise ValueError(f"page rotation {page.rotation} is not supported")
    paths = page.get_drawings()
    reach = []
    for path in paths:
        half = path["width"] / 2 if "s" in path["type"] else 0.0
        reach.append(path["rect"] + (-half, -half, half, half))
    first = [i for i, r in enumerate(reach) if r.intersects(region)]
    boxes = shapely.box(*np.array([tuple(r) for r in reach]).T)
    ends = shapely.points([pt for i in first for pt in path_points(paths[i])])
    _, near = STRtree(boxes).query(ends, predicate="dwithin", distance=JOIN_TOL)
    kept = sorted(set(first) | set(near.tolist()))
    keep = set(kept)
    remove_items(
        page, paths, [i not in keep for i in range(len(paths))], drop_text=False
    )
    height = page.mediabox.height  # mediabox is in PDF space (y up)
    page.set_mediabox(
        pymupdf.Rect(region.x0, height - region.y1, region.x1, height - region.y0)
    )
    return doc, kept


def check_same_groups(selected: list[dict], crop_groups: list[dict], kept: list[int]) -> None:
    """Raise unless each selected page group is exactly one crop group (mapped to page indices)."""
    mapped = [[kept[i] for i in g["items"]] for g in crop_groups]
    for group in selected:
        if mapped.count(group["items"]) != 1:
            same_text = [m for m, g in zip(mapped, crop_groups) if g["text"] == group["text"]]
            raise RuntimeError(
                f"{group['text']!r}: crop groups {same_text} differ from page group {group['items']}"
            )


def draw_review(pdf: Path, groups: list[dict], out: Path) -> None:
    """Render pdf with each group's items, rect, leader_rect and dot overlaid, labeled with its text."""
    doc = pymupdf.open(pdf)
    page = doc[0]
    paths = page.get_drawings()
    for n, group in enumerate(groups):
        color = COLORS[n % len(COLORS)]
        shape = page.new_shape()
        for i in group["items"]:
            for item in paths[i]["items"]:
                if item[0] == "l":
                    shape.draw_line(item[1], item[2])
                elif item[0] == "c":
                    shape.draw_bezier(*item[1:])
                elif item[0] == "re":
                    shape.draw_rect(item[1])
                elif item[0] == "qu":
                    shape.draw_quad(item[1])
                else:
                    raise ValueError("unhandled drawing", item)
            shape.finish(color=color, width=0.8)
        if isinstance(group["rect"], Polygon):
            shape.draw_polyline(list(group["rect"].exterior.coords))
        else:
            shape.draw_rect(pymupdf.Rect(group["rect"]))
        shape.finish(color=color, width=0.4, dashes="[2 1] 0")
        if group.get("leader_rect") is not None:
            shape.draw_rect(pymupdf.Rect(group["leader_rect"]))
            shape.finish(color=color, width=0.4, dashes="[0.4 0.8] 0")
        if group.get("dot") is not None:
            shape.draw_circle((group["dot"].x, group["dot"].y), 4.0)
            shape.finish(color=color, width=0.4)
        x0, y0 = rect_bounds(group["rect"]).tl
        shape.insert_text(
            (x0 + 1, y0 + 5), f"{n}: {group['text']}", fontsize=5, color=color
        )
        shape.commit()
    page.get_pixmap(dpi=400).save(str(out))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kind", choices=["measurements", "tags", "align", "callouts", "rooms"], required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    def groups_of(fp: FloorPlanParser) -> tuple[list[dict], list[dict]]:
        """(groups, unclassified) of fp for args.kind."""
        if args.kind == "measurements":
            fp.get_measurements()
            return fp.measurements, fp.unclassified_measurements
        if args.kind == "tags":
            fp.get_tags()
            return fp.tags, fp.unclassified_tags
        if args.kind == "align":
            fp.get_align_markers()
            return fp.align_markers, fp.unclassified_align_markers
        if args.kind == "callouts":
            fp.get_callouts()
            return fp.callouts, fp.unclassified_callouts
        fp.get_room_labels()
        return fp.enc_labels, fp.unclassified_room_labels

    source = pymupdf.open(SOURCE)
    full_groups, _ = groups_of(FloorPlanParser(source[0]))
    FIXTURES.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    crops = {
        "measurements": CROPS,
        "tags": TAG_CROPS,
        "align": ALIGN_CROPS,
        "callouts": CALLOUT_CROPS,
        "rooms": ROOM_CROPS,
    }[args.kind]
    for stem, picks in crops.items():
        if args.kind == "measurements":
            selected = select_measurements(full_groups, picks)
        else:
            selected = select_tags(full_groups, picks)
        region = crop_region(selected)
        pdf = FIXTURES / f"{stem}.pdf"
        doc, kept = make_crop(source, region)
        doc.save(pdf, garbage=3, deflate=True)

        crop = FloorPlanParser(pymupdf.open(pdf)[0])
        groups, unclassified = groups_of(crop)
        check_same_groups(selected, groups, kept)
        out = EXPECTED / stem
        out.mkdir(parents=True, exist_ok=True)
        if args.kind == "rooms":
            (out / "room_labels.json").write_text(json.dumps([room_label_json(g) for g in groups], indent=1) + "\n")
        else:
            (out / f"{args.kind}.json").write_text(json.dumps(groups, indent=1) + "\n")
        draw_review(pdf, groups, args.review_dir / f"{stem}.png")
        print(
            f"{stem}: region {tuple(round(v, 2) for v in region)}, {len(crop.paths)} drawing items, "
            f"{pdf.stat().st_size // 1024} KiB, groups {[g['text'] for g in groups]}, "
            f"unclassified {unclassified}"
        )


if __name__ == "__main__":
    main()
