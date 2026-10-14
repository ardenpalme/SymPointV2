"""Golden tests for FloorPlanParser.get_tags on reviewed crops of A-1-03-10.

Fixtures and expected outputs come from `python -m tests.make_fixtures --kind tags` and were
approved by visual review. Never edit tests/expected/ to make these pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from data_eng.parser import FloorPlanParser

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = Path(__file__).parent / "expected"
MIN_IOU = 0.95
CROPS = sorted(p.parent.name for p in EXPECTED.glob("*/tags.json"))


def iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two [x0, y0, x1, y1] rects."""
    ra, rb = pymupdf.Rect(a), pymupdf.Rect(b)
    inter = ra & rb
    inter_area = inter.width * inter.height if not inter.is_empty else 0.0
    union = ra.width * ra.height + rb.width * rb.height - inter_area
    return inter_area / union


def test_fixtures_exist() -> None:
    assert CROPS == [
        "tag_bent_f23",
        "tag_stacked_j02",
        "tag_straight_g03",
        "tag_touching_g03_j31",
    ]


@pytest.mark.parametrize("crop", CROPS)
def test_tag_groups_match_expected(crop: str) -> None:
    expected = json.loads((EXPECTED / crop / "tags.json").read_text())
    parser = FloorPlanParser(pymupdf.open(FIXTURES / f"{crop}.pdf")[0])
    parser.get_tags()
    got = parser.tags

    assert [g["text"] for g in got] == [g["text"] for g in expected]
    for g, e in zip(got, expected):
        assert g["items"] == e["items"], g["text"]
        assert iou(g["rect"], e["rect"]) >= MIN_IOU, g["text"]
