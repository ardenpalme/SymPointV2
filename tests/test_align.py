"""Tests for ALIGN marker grouping: clip_polygon on hand-made clip paths, and golden tests for
FloorPlanParser.get_align_markers on reviewed crops of A-1-03-10.

Fixtures and expected outputs come from `python -m tests.make_fixtures --kind align` and were
approved by visual review. Never edit tests/expected/ to make these pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from data_eng.parser import FloorPlanParser, clip_polygon
from tests.test_tags import iou

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = Path(__file__).parent / "expected"
MIN_IOU = 0.95
CROPS = sorted(p.parent.name for p in EXPECTED.glob("*/align.json"))


def lines(*points: tuple[float, float]) -> list[tuple]:
    """get_drawings()-style l items joining consecutive points."""
    return [
        ("l", pymupdf.Point(points[i]), pymupdf.Point(points[i + 1]))
        for i in range(len(points) - 1)
    ]


def test_clip_polygon_closed_triangle() -> None:
    poly = clip_polygon(lines((0, 0), (2, 0), (1, 9), (0, 0)))
    assert poly.geom_type == "Polygon"
    assert np.isclose(poly.area, 9.0)


def test_clip_polygon_drops_sliver_subpaths() -> None:
    # Diagonal arrowheads: tiny stair-step triangles at the tip, then the open main triangle.
    sliver = lines((1.0, 9.0), (1.12, 9.12), (1.12, 9.0), (1.0, 9.0))
    main = lines((0, 0), (2, 0), (1, 9))  # no closing segment
    poly = clip_polygon(sliver + main)
    assert poly.geom_type == "Polygon"
    assert np.isclose(poly.area, 9.0)


def test_clip_polygon_rejects_curves() -> None:
    curve = [("c", *(pymupdf.Point(x, 0) for x in range(4)))]
    with pytest.raises(ValueError, match="expected only l"):
        clip_polygon(lines((0, 0), (2, 0), (1, 9)) + curve)


def test_clip_polygon_keeps_two_shapes_apart() -> None:
    area = clip_polygon(
        lines((0, 0), (2, 0), (1, 9)) + lines((10, 0), (12, 0), (11, 9))
    )
    assert area.geom_type == "MultiPolygon"
    assert np.isclose(area.area, 18.0)


def test_fixtures_exist() -> None:
    assert CROPS == [
        "align_diagonal",
        "align_horizontal",
        "align_long",
        "align_side_by_side",
        "align_skewed",
        "align_vertical",
    ]


@pytest.mark.parametrize("crop", CROPS)
def test_align_groups_match_expected(crop: str) -> None:
    expected = json.loads((EXPECTED / crop / "align.json").read_text())
    parser = FloorPlanParser(pymupdf.open(FIXTURES / f"{crop}.pdf")[0])
    parser.get_align_markers()
    got = parser.align_markers

    assert parser.unclassified_align_markers == []
    assert [g["text"] for g in got] == [g["text"] for g in expected]
    for g, e in zip(got, expected):
        assert g["items"] == e["items"], g["text"]
        assert iou(g["rect"], e["rect"]) >= MIN_IOU, g["text"]
