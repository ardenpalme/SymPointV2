"""Visual check for a cleaned floor plan drawing.

Renders the source page and the cleaned drawing over the same region and writes:
  overview.png        whole region at low DPI
  tile_rRR_cCC.png    high-DPI tiles for close inspection (blank tiles are skipped)
The tiles show the cleaned drawing in grayscale, with red = removed by cleaning and
blue = darker than before: content revealed where a removed item was drawn over a fill.

Exits 1 if the cleaned drawing still contains extractable text or changed its mediabox or rotation.
Text drawn as vector strokes (e.g. CAD SHX fonts) is NOT detected here: inspect the tiles.

Usage, from the repo root:
    python -m data_eng.check_render SOURCE.pdf CLEANED.pdf --out DIR
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pymupdf

DIFF_THRESHOLD = 60  # grayscale 0-255; smaller brightness changes are antialiasing noise
ADDED_INK_TOLERANCE = 1e-4  # fraction of pixels; darkening below this is not reported


def render_gray(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    return arr[:, : pix.width]


def compare(src_page: pymupdf.Page, cln_page: pymupdf.Page, dpi: int) -> tuple[np.ndarray, float]:
    """Return an RGB overlay and the fraction of pixels where cleaning added ink.

    Compares brightness, not ink masks, so a tick removed from a filled wall still shows as red.
    """
    src = render_gray(src_page, dpi).astype(np.int16)
    cln = render_gray(cln_page, dpi).astype(np.int16)
    if src.shape != cln.shape:
        raise ValueError(f"render sizes differ at {dpi} dpi: {src.shape} vs {cln.shape}")
    removed = (cln - src) > DIFF_THRESHOLD
    added = (src - cln) > DIFF_THRESHOLD
    rgb = np.repeat(cln.astype(np.uint8)[..., None], 3, axis=2)  # cleaned drawing in grayscale
    rgb[removed] = (220, 0, 0)
    rgb[added] = (0, 0, 220)
    return rgb, float(added.mean())


def save_png(rgb: np.ndarray, path: Path) -> None:
    h, w, _ = rgb.shape
    pix = pymupdf.Pixmap(pymupdf.csRGB, w, h, np.ascontiguousarray(rgb).tobytes(), False)
    pix.save(str(path))


def save_tiles(rgb: np.ndarray, out: Path, tile_px: int) -> int:
    h, w, _ = rgb.shape
    saved = 0
    for r in range(math.ceil(h / tile_px)):
        for c in range(math.ceil(w / tile_px)):
            tile = rgb[r * tile_px : (r + 1) * tile_px, c * tile_px : (c + 1) * tile_px]
            if (tile != 255).any():
                save_png(tile, out / f"tile_r{r:02d}_c{c:02d}.png")
                saved += 1
    return saved


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("cleaned", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--page", type=int, default=0, help="source page index")
    p.add_argument("--overview-dpi", type=int, default=40)
    p.add_argument("--tile-dpi", type=int, default=150)
    p.add_argument("--tile-px", type=int, default=1200)
    args = p.parse_args(argv)

    src_doc, cln_doc = pymupdf.open(args.source), pymupdf.open(args.cleaned)
    src_page, cln_page = src_doc[args.page], cln_doc[0]
    if src_page.mediabox != cln_page.mediabox or src_page.rotation != cln_page.rotation:
        print("FAIL: cleaned page must keep the source mediabox and rotation; crop with a cropbox.")
        return 1
    src_page.set_cropbox(cln_page.cropbox)  # in memory only; the source file is never saved

    args.out.mkdir(parents=True, exist_ok=True)
    save_png(compare(src_page, cln_page, args.overview_dpi)[0], args.out / "overview.png")
    detail, added = compare(src_page, cln_page, args.tile_dpi)
    n_tiles = save_tiles(detail, args.out, args.tile_px)

    failed = False
    words = cln_page.get_text("words", clip=cln_page.rect)
    if words:
        failed = True
        print(f"FAIL: {len(words)} text words remain, e.g. {[w[4] for w in words[:20]]}")
    if added > ADDED_INK_TOLERANCE:
        # Deleting drawing items can't add ink; darker pixels are fills revealed under removed items.
        print(f"INFO: {added:.3%} of pixels got darker (blue): content revealed under removed items")
    print(f"Wrote overview.png and {n_tiles} non-blank tiles to {args.out}")
    print("FAIL" if failed else "PASS (automatic checks only; inspect the tiles)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
