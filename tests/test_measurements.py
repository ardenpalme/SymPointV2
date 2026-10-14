"""Golden tests for FloorPlanParser.get_measurements on reviewed crops of A-1-03-10.

Fixtures and expected outputs come from tests/make_fixtures.py and were approved by visual review.
Never edit tests/expected/ to make these pass.
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
CROPS = sorted(p.parent.name for p in EXPECTED.glob("*/measurements.json"))


def iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two [x0, y0, x1, y1] rects."""
    ra, rb = pymupdf.Rect(a), pymupdf.Rect(b)
    inter = ra & rb
    inter_area = inter.width * inter.height if not inter.is_empty else 0.0
    union = ra.width * ra.height + rb.width * rb.height - inter_area
    return inter_area / union


def test_fixtures_exist() -> None:
    assert CROPS == ["along_wall_2950", "chain_10030_220", "dim_1691", "leader_580"]


@pytest.mark.parametrize("crop", CROPS)
def test_measurement_groups_match_expected(crop: str) -> None:
    expected = json.loads((EXPECTED / crop / "measurements.json").read_text())
    parser = FloorPlanParser(pymupdf.open(FIXTURES / f"{crop}.pdf")[0])
    parser.get_measurements()
    got = parser.measurements

    assert [g["text"] for g in got] == [g["text"] for g in expected]
    for g, e in zip(got, expected):
        assert g["items"] == e["items"], g["text"]
        assert iou(g["rect"], e["rect"]) >= MIN_IOU, g["text"]
        if e["leader_rect"] is None:
            assert g["leader_rect"] is None, g["text"]
        else:
            assert iou(g["leader_rect"], e["leader_rect"]) >= MIN_IOU, g["text"]
