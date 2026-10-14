from __future__ import annotations

from collections.abc import Sequence

import pymupdf
from pymupdf import Rect, mupdf
from shapely import LineString, MultiPolygon, Point, Polygon

BBOX_TOL = 0.01  # pt; max gap between a culler bbox and its get_drawings() rect

def draw_geoms(geoms, shape, dot_radius=1):
    for prim in geoms:
        if isinstance(prim, Polygon):
            shape.draw_polyline(list(prim.exterior.coords))
        elif isinstance(prim, MultiPolygon):
            for part in prim.geoms:
                shape.draw_polyline(list(part.exterior.coords))
        elif isinstance(prim, LineString):
            shape.draw_polyline(list(prim.coords))
        elif isinstance(prim, Point):
            shape.draw_circle(next(iter(prim.coords)), dot_radius)
        else:
            print("svg primitive unhandled", type(prim))
            break

def display_geoms(page, geoms, fname, show_bg=True, crop=None):
    out = pymupdf.open()
    if(crop != None): 
        minx, miny, maxx, maxy = crop.bounds
        clip = Rect([(minx, miny), (maxx, maxy)])
        outpage = out.new_page(width=maxx - minx, height=maxy - miny)
        if(show_bg): outpage.show_pdf_page(outpage.rect, page.parent, page.number, clip=clip)
    else:
        outpage = out.new_page(width=page.rect.width, height=page.rect.height)
        if(show_bg): outpage.show_pdf_page(outpage.rect, page.parent, page.number)
            
    shape = outpage.new_shape()
    draw_geoms(geoms, shape)
    shape.finish(color=(1, 0, 0), width=0.5)
    shape.commit()
    out.save(fname)


def remove_items(
    page: pymupdf.Page, paths: list[dict], drop_path: Sequence[bool], drop_text: bool
) -> None:
    """Delete all text (if drop_text) and the paths flagged in drop_path from page.

    paths is page.get_drawings() of an identical page and drop_path is aligned with it.
    MuPDF's sanitize filter calls the culler once per fill and once per stroke, in content
    order, with only its bbox: the k-th fill call is the k-th path whose type contains "f"
    and the k-th stroke call the k-th path whose type contains "s" (an "fs" path gets one
    of each). Fill calls are checked against the path's rect, stroke calls against the rect
    inflated by width/2 (page space, pt). Dropping an "fs" path drops both its fill and stroke.
    """
    if len(drop_path) != len(paths):
        raise ValueError(f"drop_path has {len(drop_path)} entries for {len(paths)} paths")
    # Paths in content order per culler kind: "f" gets fill calls, "s" stroke calls.
    queues = {kind: [(p, d) for p, d in zip(paths, drop_path) if kind in p["type"]] for kind in "fs"}
    call_kinds = {
        mupdf.FZ_CULL_PATH_FILL: "f",
        mupdf.FZ_CULL_PATH_STROKE: "s",
        mupdf.FZ_CULL_PATH_FILL_STROKE: "fs",  # one call for a "B" operator
    }
    to_page = page.transformation_matrix  # culler bboxes are in PDF space (y up)
    calls = {kind: 0 for kind in "fs"}
    # Exceptions raised inside the culler are swallowed by MuPDF, so record the first
    # mismatch, keep everything from then on, and raise after filtering.
    errors: list[str] = []

    class Culler(mupdf.PdfSanitizeFilterOptions2):
        def __init__(self) -> None:
            super().__init__()
            self.use_virtual_culler()

        def culler(self, ctx, bbox, cull_type):
            if cull_type == mupdf.FZ_CULL_GLYPH:
                return int(drop_text)
            kinds = call_kinds.get(cull_type)
            if kinds is None:
                return 0
            matched = []
            for kind in kinds:
                calls[kind] += 1
                if calls[kind] <= len(queues[kind]):
                    matched.append(queues[kind][calls[kind] - 1])
            if errors:
                return 0
            if len(matched) != len(kinds) or any(m[0] is not matched[0][0] for m in matched):
                errors.append(f"culler call {kinds!r} #{calls} has no matching path")
                return 0
            path, drop = matched[0]
            got = pymupdf.Rect(mupdf.FzRect(bbox)) * to_page
            half = path["width"] / 2 if "s" in kinds else 0.0
            want = path["rect"] + (-half, -half, half, half)
            if max(abs(a - b) for a, b in zip(got, want)) > BBOX_TOL:
                errors.append(f"culler call {kinds!r} #{calls}: bbox {got} does not match path {want}")
                return 0
            return int(drop)

    culler_opts = Culler()

    class Sanitize(mupdf.PdfFilterFactory2):
        def __init__(self) -> None:
            super().__init__()
            self.use_virtual_filter()

        def filter(self, ctx, doc, chain, struct_parents, transform, options):
            return mupdf.ll_pdf_new_sanitize_filter(
                doc, chain, struct_parents, transform, options, culler_opts.internal()
            )

    factory = Sanitize()
    opts = mupdf.PdfFilterOptions()
    opts.recurse = 1
    opts.add_factory(factory.internal())
    pdf_page = mupdf.pdf_page_from_fz_page(page.this)
    mupdf.pdf_filter_page_contents(pdf_page.doc(), pdf_page, opts)
    if errors:
        raise RuntimeError(f"path order mismatch, page left unusable: {errors[0]}")
    for kind in "fs":
        if calls[kind] != len(queues[kind]):
            raise RuntimeError(f"{calls[kind]} {kind!r} culler calls for {len(queues[kind])} paths")
