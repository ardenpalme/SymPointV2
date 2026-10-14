"""Build the measurement fixtures: small crop PDFs of A-1-03-10 and candidate expected groups.

Usage, from the repo root:
    python -m tests.make_fixtures --review-dir DIR

Writes tests/fixtures/<crop>.pdf, tests/expected/<crop>/measurements.json, and DIR/<crop>.png
with the candidate groups overlaid (one color per group, rect dashed, leader_rect dotted).
The JSON files are candidates: they are produced by get_measurements() itself and become
expected outputs only after a human has reviewed the PNGs. Rerunning overwrites them.

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
from shapely import STRtree

from data_eng.parser import JOIN_TOL, FloorPlanParser
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
COLORS = [
    (0.9, 0, 0),
    (0, 0.6, 0),
    (0, 0, 0.9),
    (0.9, 0.5, 0),
    (0.6, 0, 0.8),
    (0, 0.6, 0.6),
]


def crop_region(groups: list[dict], texts: list[str]) -> pymupdf.Rect:
    """Union of the rects of the groups with these texts, plus CROP_MARGIN (page space, pt)."""
    region = pymupdf.Rect()
    for text in texts:
        matches = [g for g in groups if g["text"] == text]
        if len(matches) != 1:
            raise ValueError(
                f"{text!r} has {len(matches)} measurement groups on the page, expected 1"
            )
        region |= pymupdf.Rect(matches[0]["rect"])
        if matches[0]["leader_rect"] is not None:
            region |= pymupdf.Rect(matches[0]["leader_rect"])
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


def check_same_groups(
    texts: list[str], page_groups: list[dict], crop_groups: list[dict], kept: list[int]
) -> None:
    """Raise if a crop's groups for texts, mapped to page indices, differ from the page's."""
    for text in texts:
        want = [g["items"] for g in page_groups if g["text"] == text]
        got = [[kept[i] for i in g["items"]] for g in crop_groups if g["text"] == text]
        if got != want:
            raise RuntimeError(
                f"{text!r}: crop groups {got} differ from page groups {want}"
            )


def draw_review(pdf: Path, groups: list[dict], out: Path) -> None:
    """Render pdf with each group's items, rect and leader_rect overlaid, labeled with its text."""
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
        shape.draw_rect(pymupdf.Rect(group["rect"]))
        shape.finish(color=color, width=0.4, dashes="[2 1] 0")
        if group["leader_rect"] is not None:
            shape.draw_rect(pymupdf.Rect(group["leader_rect"]))
            shape.finish(color=color, width=0.4, dashes="[0.4 0.8] 0")
        x0, y0 = group["rect"][:2]
        shape.insert_text(
            (x0 + 1, y0 + 5), f"{n}: {group['text']}", fontsize=5, color=color
        )
        shape.commit()
    page.get_pixmap(dpi=400).save(str(out))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    source = pymupdf.open(SOURCE)
    full = FloorPlanParser(source[0])
    full.get_measurements()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    for stem, texts in CROPS.items():
        region = crop_region(full.measurements, texts)
        pdf = FIXTURES / f"{stem}.pdf"
        doc, kept = make_crop(source, region)
        doc.save(pdf, garbage=3, deflate=True)

        crop = FloorPlanParser(pymupdf.open(pdf)[0])
        crop.get_measurements()
        check_same_groups(texts, full.measurements, crop.measurements, kept)
        out = EXPECTED / stem
        out.mkdir(parents=True, exist_ok=True)
        (out / "measurements.json").write_text(
            json.dumps(crop.measurements, indent=1) + "\n"
        )
        draw_review(pdf, crop.measurements, args.review_dir / f"{stem}.png")
        print(
            f"{stem}: region {tuple(round(v, 2) for v in region)}, {len(crop.paths)} drawing items, "
            f"{pdf.stat().st_size // 1024} KiB, groups {[g['text'] for g in crop.measurements]}, "
            f"unclassified {crop.unclassified_measurements}"
        )


if __name__ == "__main__":
    main()
