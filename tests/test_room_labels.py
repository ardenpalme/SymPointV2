"""Golden tests for FloorPlanParser.get_room_labels on reviewed crops of A-1-03-10.

Fixtures and expected outputs come from `python -m tests.make_fixtures --kind rooms` and were
approved by visual review. Never edit tests/expected/ to make these pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from shapely import Polygon

from data_eng.parser import JOIN_TOL, FloorPlanParser, room_label_json

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = Path(__file__).parent / "expected"
MIN_IOU = 0.95
CROPS = sorted(p.parent.name for p in EXPECTED.glob("*/room_labels.json"))


def polygon_iou(a: list[list[float]], b: list[list[float]]) -> float:
    """Intersection over union of two polygons given as [x, y] corners."""
    pa, pb = Polygon(a), Polygon(b)
    return pa.intersection(pb).area / pa.union(pb).area


def test_fixtures_exist() -> None:
    assert CROPS == [
        "room_bent_leader_03-C15",
        "room_leader_crosses_walls_03-C17",
        "room_no_leader_03-A20",
        "room_no_leader_03-A55",
        "room_straight_leader_03-A36",
    ]


@pytest.mark.parametrize("crop", CROPS)
def test_room_label_groups_match_expected(crop: str) -> None:
    expected = json.loads((EXPECTED / crop / "room_labels.json").read_text())
    parser = FloorPlanParser(pymupdf.open(FIXTURES / f"{crop}.pdf")[0])
    parser.get_room_labels()
    got = [room_label_json(g) for g in parser.enc_labels]

    assert parser.unclassified_room_labels == []
    assert [(g["text"], g["name"]) for g in got] == [
        (g["text"], g["name"]) for g in expected
    ]
    for g, e in zip(got, expected):
        assert g["items"] == e["items"], g["text"]
        assert polygon_iou(g["rect"], e["rect"]) >= MIN_IOU, g["text"]
        if e["leader_rect"] is None:
            assert g["leader_rect"] is None, g["text"]
            assert g["dot"] is None, g["text"]
        else:
            # A straight leader along an axis has a zero-area rect, so compare corners, not IoU.
            assert np.allclose(g["leader_rect"], e["leader_rect"], atol=JOIN_TOL), g[
                "text"
            ]
            assert math.dist(g["dot"], e["dot"]) <= JOIN_TOL, g["text"]
