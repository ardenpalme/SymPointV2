"""Golden tests for FloorPlanParser.get_oval_tags on reviewed crops of A-1-03-10.

Fixtures and expected outputs come from `python -m tests.make_fixtures --kind ovals` and were
approved by visual review. Never edit tests/expected/ to make these pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from shapely import Polygon

from data_eng.parser import JOIN_TOL, FloorPlanParser, oval_tag_json

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = Path(__file__).parent / "expected"
MIN_IOU = 0.95
CROPS = sorted(p.parent.name for p in EXPECTED.glob("*/oval_tags.json"))


def polygon_iou(a: list[list[float]], b: list[list[float]]) -> float:
    """Intersection over union of two polygons given as [x, y] corners."""
    pa, pb = Polygon(a), Polygon(b)
    return pa.intersection(pb).area / pa.union(pb).area


def test_fixtures_exist() -> None:
    assert CROPS == ["oval_bent_leader_110", "oval_bent_leader_111", "oval_window_110"]


def test_requires_room_labels() -> None:
    parser = FloorPlanParser(pymupdf.open(FIXTURES / "oval_bent_leader_110.pdf")[0])
    with pytest.raises(RuntimeError, match="get_room_labels"):
        parser.get_oval_tags()


@pytest.mark.parametrize("crop", CROPS)
def test_oval_tag_groups_match_expected(crop: str) -> None:
    expected = json.loads((EXPECTED / crop / "oval_tags.json").read_text())
    parser = FloorPlanParser(pymupdf.open(FIXTURES / f"{crop}.pdf")[0])
    parser.get_room_labels()
    parser.get_oval_tags()
    got = [oval_tag_json(g) for g in parser.enc_labels if g["kind"] == "oval_tag"]

    assert parser.unclassified_oval_tags == []
    assert [g["text"] for g in got] == [g["text"] for g in expected]
    for g, e in zip(got, expected):
        assert g["items"] == e["items"], g["text"]
        assert polygon_iou(g["rect"], e["rect"]) >= MIN_IOU, g["text"]
        # A straight leader along an axis has a zero-area rect, so compare corners, not IoU.
        assert np.allclose(g["leader_rect"], e["leader_rect"], atol=JOIN_TOL), g["text"]
