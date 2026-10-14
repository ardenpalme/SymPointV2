from __future__ import annotations

import pymupdf
import pytest

from data_eng.utils import remove_items


def make_page() -> pymupdf.Page:
    """One page with, in content order: a stroke, a fill, a fill+stroke, and text."""
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    shape = page.new_shape()
    shape.draw_line((10, 10), (90, 10))
    shape.finish(color=(0, 0, 0), width=0.5)
    shape.draw_rect(pymupdf.Rect(10, 30, 50, 60))
    shape.finish(color=None, fill=(1, 1, 1))
    shape.draw_rect(pymupdf.Rect(60, 30, 100, 60))
    shape.finish(color=(0, 0, 0), fill=(0, 0, 1), width=1.0)
    shape.insert_text((10, 100), "1691 mm")
    shape.commit()
    return page


def types_after_drop(drop: list[bool]) -> list[str]:
    page = make_page()
    paths = page.get_drawings()
    assert [p["type"] for p in paths] == ["s", "f", "fs"]
    remove_items(page, paths, drop, drop_text=False)
    return [p["type"] for p in page.get_drawings()]


def test_drops_fill_only_path() -> None:
    assert types_after_drop([False, True, False]) == ["s", "fs"]


def test_drops_fill_and_stroke_path() -> None:
    assert types_after_drop([False, False, True]) == ["s", "f"]


def test_drops_stroke_only_path() -> None:
    assert types_after_drop([True, False, False]) == ["f", "fs"]


def test_keeps_text_unless_asked() -> None:
    page = make_page()
    remove_items(page, page.get_drawings(), [False, False, False], drop_text=False)
    assert page.get_text().strip() == "1691 mm"


def test_rejects_misaligned_drop_flags() -> None:
    page = make_page()
    with pytest.raises(ValueError, match="2 entries for 3 paths"):
        remove_items(page, page.get_drawings(), [True, False], drop_text=False)
