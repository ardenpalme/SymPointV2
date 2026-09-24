"""
PDF -> SymPointV2 npz, matched to FloorPlanCAD's parse_svg.py encoding.

Fixes vs. the original parse_pdf.py
  1. Arcs: CAD PDF exporters tessellate arcs/circles into many short lines.
     Connected short-line chains with steady turning are refitted to a circle and
     emitted as ONE Arc primitive (cmd 1), or ONE circle (cmd 2) if closed.
     Consecutive Bezier curves forming one arc/circle are merged the same way.
  2. Layers: OCG name if present, else pseudo-layer = (stroke color, width), which is
     how CAD exports usually differ by layer. IDs start at 1 (no collision).
  3. Widths: constant 1.0, as in FloorPlanCAD after per-sample normalization.

Primitive encoding (same as parse_svg.py): 4 points sampled at 0, 1/3, 2/3, 1 of the
length for lines/arcs; circle at angles 0, pi/2, pi, 3pi/2; length in page units.
"""
import argparse, glob, math, os
from collections import defaultdict
import numpy as np

LABEL_NUM = 35
CMD_LINE, CMD_ARC, CMD_CIRCLE = 0, 1, 2
TS = (0.0, 1 / 3, 2 / 3, 1.0)

# arc reconstruction thresholds
JOIN_TOL = 0.05        # endpoint snapping tolerance (PDF points)
MIN_TURN = math.radians(0.5)
MAX_TURN = math.radians(35)
MIN_SEGS = 3           # tessellated arcs have >= 3 segments
MAX_SEG_RATIO = 3.0    # segment lengths within a run must be similar
FIT_TOL = 0.03         # max radial residual / r
MIN_ARC_ANGLE = math.radians(10)


# ---------------------------------------------------------------- geometry
def polyline_samples(P, ts=TS):
    """Points at fractions ts of the arc length of polyline P (k,2)."""
    d = np.hypot(*np.diff(P, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    L = cum[-1]
    out = []
    for t in ts:
        s = t * L
        i = min(np.searchsorted(cum, s, side="right") - 1, len(d) - 1)
        f = 0.0 if d[i] == 0 else (s - cum[i]) / d[i]
        out.append(P[i] + f * (P[i + 1] - P[i]))
    return np.array(out), float(L)


def fit_circle(P):
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(P))]
    (cx, cy, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    if r == 0:
        return cx, cy, r, np.inf
    return cx, cy, r, float(np.abs(np.hypot(x - cx, y - cy) - r).max() / r)


def bezier_pts(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    u = 1 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- primitives
def prim_line(a, b):
    P = np.array([a, b], float)
    pts, L = polyline_samples(P)
    return CMD_LINE, pts, L


def prim_arc(P):
    pts, L = polyline_samples(P)
    return CMD_ARC, pts, L


def prim_circle(cx, cy, r):
    th = np.array([0, math.pi / 2, math.pi, 3 * math.pi / 2])
    return CMD_CIRCLE, np.c_[cx + r * np.cos(th), cy + r * np.sin(th)], 2 * math.pi * r


def arc_or_none(P, closed):
    """P: ordered vertices of a curved run. Returns a primitive or None."""
    cx, cy, r, err = fit_circle(P)
    if err > FIT_TOL:
        return None
    ang = np.unwrap(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
    sweep = abs(ang[-1] - ang[0])
    if closed and abs(sweep - 2 * math.pi) < math.radians(20):
        return prim_circle(cx, cy, r)
    if sweep < MIN_ARC_ANGLE:
        return None
    return prim_arc(P)


# ---------------------------------------------------------------- line chains
def chain_segments(segs):
    """segs: (M,2,2) same-layer line segments. Returns list of (vertex array, seg ids, closed)."""
    q = lambda p: (round(p[0] / JOIN_TOL), round(p[1] / JOIN_TOL))
    node = defaultdict(list)
    for i, (a, b) in enumerate(segs):
        node[q(a)].append((i, 0))
        node[q(b)].append((i, 1))
    used = np.zeros(len(segs), bool)
    chains = []

    def walk(seg, end):
        """Follow from segment `seg` leaving through endpoint `end`; return [(seg, end_out)...]."""
        out = []
        while True:
            nk = q(segs[seg][end])
            nb = [(s, e) for s, e in node[nk] if s != seg]
            if len(node[nk]) != 2 or not nb or used[nb[0][0]]:
                return out, False
            s, e = nb[0]
            used[s] = True
            out.append((s, 1 - e))          # enter at e, leave at 1-e
            seg, end = s, 1 - e

    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        fwd, _ = walk(i, 1)
        bwd, _ = walk(i, 0)
        order = [(s, 1 - e) for s, e in reversed(bwd)] + [(i, 1)] + fwd  # (seg, exit end)
        V = [segs[order[0][0]][1 - order[0][1]]] + [segs[s][e] for s, e in order]
        V = np.array(V, float)
        closed = len(order) > 2 and q(V[0]) == q(V[-1])
        chains.append((V, [s for s, _ in order], closed))
    return chains


def chain_to_prims(V, closed):
    """Split a vertex chain into curved runs (-> arc/circle) and plain lines."""
    k = len(V) - 1                                  # segments
    d = np.hypot(*np.diff(V, axis=0).T)
    h = np.arctan2(*np.diff(V, axis=0).T[::-1])
    turns = np.array([wrap(h[i + 1] - h[i]) for i in range(k - 1)])

    if closed and k >= 4:
        tc = np.append(turns, wrap(h[0] - h[-1]))
        if (np.all((np.abs(tc) > MIN_TURN) & (np.abs(tc) < MAX_TURN))
                and (np.all(tc > 0) or np.all(tc < 0))
                and d.max() / max(d.min(), 1e-9) < MAX_SEG_RATIO):
            p = arc_or_none(V, closed=True)
            if p is not None:
                return [p]

    prims, i = [], 0                                # i = segment index
    while i < k:
        j = i                                       # grow run of segments i..j
        while j + 1 < k:
            t = turns[j]
            ok = MIN_TURN < abs(t) < MAX_TURN
            ok &= (j == i) or (np.sign(t) == np.sign(turns[i]))
            seg = d[i:j + 2]
            ok &= seg.max() / max(seg.min(), 1e-9) < MAX_SEG_RATIO
            if not ok:
                break
            j += 1
        p = None
        if j - i + 1 >= MIN_SEGS:
            p = arc_or_none(V[i:j + 2], closed=False)
        if p is not None:
            prims.append(p)
            i = j + 1
        else:
            prims.append(prim_line(V[i], V[i + 1]))
            i += 1
    return prims


# ---------------------------------------------------------------- beziers
def bezier_run_to_prims(run):
    """run: list of consecutive connected cubic Beziers [(p0,p1,p2,p3), ...]."""
    P = np.concatenate([bezier_pts(*b, 9)[(0 if n == 0 else 1):] for n, b in enumerate(run)])
    closed = np.hypot(*(P[0] - P[-1])) < JOIN_TOL
    if len(run) > 1:
        p = arc_or_none(P, closed)
        if p is not None:
            return [p]
    return [prim_arc(bezier_pts(*b, 33)) for b in run]


# ---------------------------------------------------------------- page
def parse_pdf(pdf_path, page_idx=0):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    page = doc[page_idx]
    W, H = int(page.rect.width), int(page.rect.height)

    layer_ids = {}
    prims = []                                      # (cmd, pts(4,2), length, layer, rgb)
    lines = defaultdict(list)                       # layer -> [(a, b)]
    line_rgb = {}

    for d in page.get_drawings():
        if d.get("type") == "f":
            continue
        rgb = tuple(int(round(c * 255)) for c in (d.get("color") or (0, 0, 0)))
        w = float(d.get("width") or 0.0)
        key = d.get("layer") or (rgb, round(w, 3))
        lid = layer_ids.setdefault(key, len(layer_ids) + 1)
        line_rgb[lid] = rgb

        bez_run = []
        def flush():
            if bez_run:
                prims.extend((*p, lid, rgb) for p in bezier_run_to_prims(bez_run))
                bez_run.clear()

        for item in d.get("items", []):
            op = item[0]
            if op == "c":
                b = tuple(np.array([item[k].x, item[k].y], float) for k in range(1, 5))
                if bez_run and np.hypot(*(bez_run[-1][3] - b[0])) >= JOIN_TOL:
                    flush()
                bez_run.append(b)
                continue
            flush()
            if op == "l":
                lines[lid].append(((item[1].x, item[1].y), (item[2].x, item[2].y)))
            elif op in ("re", "qu"):
                if op == "re":
                    r = item[1]
                    c = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                else:
                    qd = item[1]
                    c = [(qd.ul.x, qd.ul.y), (qd.ur.x, qd.ur.y), (qd.lr.x, qd.lr.y), (qd.ll.x, qd.ll.y)]
                for i in range(4):
                    prims.append((*prim_line(c[i], c[(i + 1) % 4]), lid, rgb))
        flush()
    doc.close()

    for lid, segs in lines.items():
        segs = np.array(segs, float)
        segs = segs[np.hypot(*(segs[:, 1] - segs[:, 0]).T) > 1e-6]   # drop zero-length
        for V, _, closed in chain_segments(segs):
            prims.extend((*p, lid, line_rgb[lid]) for p in chain_to_prims(V, closed))

    n = len(prims)
    return {
        "coords": np.array([p[1] for p in prims], np.float32).reshape(n, 8),
        "lengths": np.array([p[2] for p in prims], np.float32),
        "commands": np.array([p[0] for p in prims], np.int16),
        "semanticIds": np.full(n, LABEL_NUM, np.int16),
        "instanceIds": np.full(n, -1, np.int32),
        "rgb": np.array([p[4] for p in prims], np.uint8).reshape(n, 3),
        "widths": np.ones(n, np.float32),
        "layerIds": np.array([p[3] for p in prims], np.int16),
        "width": np.int32(W),
        "height": np.int32(H),
    }


def process(pdf_path):
    out = parse_pdf(pdf_path)
    np.savez_compressed(pdf_path.replace(".pdf", "_s2.npz"), **out)
    c = np.bincount(out["commands"], minlength=4)
    print(f"{os.path.basename(pdf_path)}: {len(out['commands'])} prims, "
          f"cmd frac {np.round(c / max(c.sum(), 1), 3)}, layers {len(np.unique(out['layerIds']))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parse PDFs for SymPointV2 inference")
    p.add_argument("--data", required=True)
    p.add_argument("--nproc", type=int, default=1)
    args = p.parse_args()
    paths = sorted(glob.glob(os.path.join(args.data, "*.pdf")))
    if args.nproc > 1:
        from multiprocessing import Pool
        with Pool(args.nproc) as pool:
            pool.map(process, paths)
    else:
        for f in paths:
            process(f)
