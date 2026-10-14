"""Tests for reference callout grouping: path_line and circle_of on hand-made paths, and golden
tests for FloorPlanParser.get_callouts on reviewed crops of A-1-03-10.

Fixtures and expected outputs come from `python -m tests.make_fixtures --kind callouts` and were
approved by visual review. Never edit tests/expected/ to make these pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf
import pytest
import shapely

from data_eng.parser import FloorPlanParser, circle_of, path_line
from tests.test_tags import iou

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = Path(__file__).parent / "expected"
MIN_IOU = 0.95
CROPS = sorted(p.parent.name for p in EXPECTED.glob("*/callouts.json"))

KAPPA = 0.5523  # Bezier control distance / radius for a quarter circle


def quarter_arcs(cx: float, cy: float, rx: float, ry: float) -> list[tuple]:
    """get_drawings()-style c items: four quarter arcs of an axis-aligned ellipse, closed."""
    items = []
    for k in range(4):
        a0, a1 = k * np.pi / 2, (k + 1) * np.pi / 2
        p0 = np.array([cx + rx * np.cos(a0), cy + ry * np.sin(a0)])
        p3 = np.array([cx + rx * np.cos(a1), cy + ry * np.sin(a1)])
        t0 = np.array([-rx * np.sin(a0), ry * np.cos(a0)]) * KAPPA
        t1 = np.array([-rx * np.sin(a1), ry * np.cos(a1)]) * KAPPA
        pts = (p0, p0 + t0, p3 - t1, p3)
        items.append(("c", *(pymupdf.Point(*p) for p in pts)))
    return items


def stroke(items: list[tuple]) -> dict:
    return {"type": "s", "items": items, "width": 0.36, "color": (0.0, 0.0, 0.0)}


def test_circle_of_finds_center_and_radius() -> None:
    found = circle_of(stroke(quarter_arcs(100.0, 50.0, 14.16, 14.16)))
    assert found is not None
    center, radius = found
    assert np.allclose(center, [100.0, 50.0], atol=0.01)
    assert np.isclose(radius, 14.16, atol=0.01)


def test_circle_of_rejects_ellipse() -> None:
    assert circle_of(stroke(quarter_arcs(0.0, 0.0, 14.0, 13.0))) is None


def test_circle_of_rejects_open_arc() -> None:
    assert circle_of(stroke(quarter_arcs(0.0, 0.0, 14.0, 14.0)[:3])) is None


def test_circle_of_rejects_square_of_curves() -> None:
    corners = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
    items = []
    for n in range(4):
        a, b = np.array(corners[n], float), np.array(corners[n + 1], float)
        pts = (a, a + (b - a) / 3, a + 2 * (b - a) / 3, b)
        items.append(("c", *(pymupdf.Point(*p) for p in pts)))
    assert circle_of(stroke(items)) is None


def test_circle_of_rejects_lines() -> None:
    items = [("l", pymupdf.Point(0, 0), pymupdf.Point(10, 0))]
    assert circle_of(stroke(items)) is None


def test_path_line_joins_lines_and_curves() -> None:
    arc = quarter_arcs(0.0, 0.0, 10.0, 10.0)[0]  # (10, 0) to (0, 10)
    items = [("l", pymupdf.Point(20, 0), pymupdf.Point(10, 0)), arc]
    line = path_line(stroke(items))
    assert np.isclose(line.length, 10.0 + np.pi * 10.0 / 2, rtol=1e-3)
    assert line.distance(shapely.Point(0, 10)) < 1e-9


def test_path_line_rejects_rects() -> None:
    items = [("re", pymupdf.Rect(0, 0, 10, 10), 1)]
    with pytest.raises(ValueError, match="unhandled drawing"):
        path_line(stroke(items))


def test_fixtures_exist() -> None:
    assert CROPS == [
        "callout_corner_one",
        "callout_corners_three",
        "callout_parts_scattered",
        "callout_pointer_one",
        "callout_pointers_two",
    ]


@pytest.mark.parametrize("crop", CROPS)
def test_callout_groups_match_expected(crop: str) -> None:
    expected = json.loads((EXPECTED / crop / "callouts.json").read_text())
    parser = FloorPlanParser(pymupdf.open(FIXTURES / f"{crop}.pdf")[0])
    parser.get_callouts()
    got = parser.callouts

    assert parser.unclassified_callouts == []
    assert [g["text"] for g in got] == [g["text"] for g in expected]
    for g, e in zip(got, expected):
        assert g["items"] == e["items"], g["text"]
        assert iou(g["rect"], e["rect"]) >= MIN_IOU, g["text"]
