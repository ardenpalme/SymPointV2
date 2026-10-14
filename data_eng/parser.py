from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pymupdf
import shapely
import yaml
from shapely import LineString, MultiLineString, MultiPolygon, Polygon, STRtree
from shapely.ops import polygonize
from tqdm import tqdm

from .geom import bezier_points, find_rectangles
from .utils import display_geoms, remove_items

THRSH = 3.0
ANGLE_TOL = 3.0

# Measurement grouping. Ranges in comments were measured in A-1-03-10 (pt = PDF points).
MEAS_TEXT_RE = re.compile(r"\d+(\.\d+)? ?mm")  # whole text of a dimension label, e.g. "1691 mm"
JOIN_TOL = 0.5  # pt; max gap between consecutive leader pieces
TICK_MAX_LEN = 7.0  # pt; ticks are 5.9-6.1 pt long
TICK_END_TOL = 0.5  # pt; ticks are drawn as two halves that start where they cross the line (0-0.12)
OVERSHOOT_MAX = 4.0  # pt; dimension lines run 2.7-2.9 pt past their end ticks, extension lines past the line
EXT_OVERSHOOT_MIN = 2.0  # pt; extension lines run 2.7-2.9 pt past the dimension line
EXT_TOL = 0.5  # pt; max distance from an extension line to its tick crossing (0-0.09)
CENTER_TOL = 0.25  # fraction of the dimension line's length; text centers are 0.00-0.14 off its middle
LEADER_MAX_SEGS = 50  # pieces in one leader chain
LEADER_MAX_TURN = 60.0  # degrees; max change of direction from one leader piece to the next
MASK_MIN_COVER = 0.9  # fraction of the text bbox area a mask covers; masks cover 0.97-0.99
MASK_MAX_AREA_RATIO = 1.75  # mask area / text bbox area; masks are 1.12-1.49

# Tag grouping. Ranges in comments were measured in A-1-03-10.
TAG_EDGE_TOL = 0.05  # pt; diamond corners come from edge ends rounded to 0.01 pt; touching tags are 0.4 apart
TAG_MASK_TOL = 0.5  # pt; max difference between a mask's bounds and its diamond's bbox (0.10-0.15)
TAG_BAR_MAX_LEN = 10.0  # pt; end bars are 5.6-5.8 pt long
TAG_LEADER_MAX_TURN = 100.0  # degrees; tag leaders bend by 45-90 (85 < max <= 90)
TAG_BAR_MIN_WIDTH_RATIO = 3.0  # end bar stroke width / leader stroke width; 1.92 / 0.36 = 5.3

# ALIGN marker grouping. Ranges in comments were measured in A-1-03-10.
ALIGN_TEXT = "ALIGN"  # whole text of the marker's label
SLIVER_MAX_AREA_FRAC = 0.05  # of a clip's area; stair steps at diagonal arrowhead tips are < 0.003
SUBPATH_TOL = 1e-3  # pt; an item starting farther than this from the previous end starts a subpath
ALIGN_TEXT_MAX_GAP = 4.0  # pt; crossbar to text bbox, 0.95-2.41
ALIGN_LEG_MAX_LEN = 20.0  # pt; legs are 1.7-14 pt long
ALIGN_HEAD_MAX_LEN = 12.0  # pt; long side of an arrowhead's min rotated rectangle, 8.7-9.1
ALIGN_HEAD_MIN_ASPECT = 2.5  # arrowhead long side / short side; 2.2-2.3 pt wide, so about 3.9

# Callout grouping. Ranges in comments were measured in A-1-03-10.
CALLOUT_ROUND_TOL = 0.2  # pt; closed curve-only paths deviate 0.013-0.055 from their radius
CALLOUT_TOUCH_TOL = 0.5  # pt; black part clips touch the circle (0.0), outlines follow them (chords about 0.11 off the arc)
CALLOUT_NUMBER_RE = re.compile(r"\d+")  # whole text of a view number next to a black part, e.g. "5"
CALLOUT_NUMBER_MAX_GAP = 4.0  # pt; black part to view number bbox, 0.0-3.05; the next text is 4.61 away

# Room label grouping. Ranges in comments were measured in A-1-03-10.
ROOM_NUMBER_RE = re.compile(r"\d\d-[A-Z]\d\d[A-Z]?")  # whole text of a room number, e.g. "03-C15"
ROOM_BOX_MAX_AREA_RATIO = 1.75  # box area / number text bbox area; boxes are 1.42-1.46
ROOM_EDGE_TOL = 0.05  # pt; box corners come from edge ends rounded to 0.01 pt
ROOM_MASK_TOL = 0.5  # pt; max difference between the number's mask bounds and its box (0.0)
ROOM_NAME_MAX_GAP = 3.0  # pt; box to first name line 1.96-2.21, between name lines 1.65-1.78
ROOM_NAME_CENTER_TOL = 1.0  # pt; name line center to box center, -0.20-0.11
ROOM_DOT_MAX_RADIUS = 4.0  # pt; end dots are 2.82-2.88
ROOM_LEADER_MAX_TURN = 90.0  # degrees; room leaders bend by 45-60

# Oval tag grouping. Ranges in comments were measured in A-1-03-10.
OVAL_NUMBER_RE = re.compile(r"\d+")  # whole text of an oval tag number, e.g. "110"
OVAL_SEG_MAX_LEN = 5.0  # pt; oval segments are 1.33-3.85 pt long
OVAL_MAX_GAP = 10.0  # pt; number bbox to oval, 2.7-7.3
OVAL_RING_TOL = 0.5  # pt; oval segments to the convex hull outline of the oval, 0.0-0.043
OVAL_MASK_TOL = 0.5  # pt; max difference between the number's mask bounds and the oval's (0.0)
OVAL_HEAD_MAX_LEN = 12.0  # pt; longest side of an arrowhead triangle, 8.5-8.67
OVAL_HEAD_MIN_ASPECT = 2.5  # arrowhead longest side / shortest side, 3.7-3.9
OVAL_LEADER_MAX_TURN = 90.0  # degrees; oval tag leaders bend by 41-62


def style_key(path: dict) -> tuple | None:
    """(color, width) of a get_drawings() path, rounded to 4 decimals; None if either is None."""
    path_width, path_color = path.get("width"), path.get("color")
    if path_width is None or path_color is None:
        return None
    return (tuple(round(ele, 4) for ele in path_color), round(path_width, 4))


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between undirected lines with angles a and b (degrees), folded into [0, 90]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


def clip_polygon(items: list) -> Polygon | MultiPolygon:
    """The area of a clip path made of l items (page space, pt): a Polygon, or a MultiPolygon
    if the clip has several separate shapes.

    A new subpath starts where an item does not start at the previous item's end. Each subpath
    is closed and made valid; polygon parts smaller than SLIVER_MAX_AREA_FRAC of the total area
    (stair steps at diagonal arrowhead tips) are dropped. Raises ValueError for items other than l.
    """
    kinds = {item[0] for item in items}
    if kinds != {"l"}:
        raise ValueError(f"clip path with items {sorted(kinds)}, expected only l")
    rings = [[tuple(items[0][1])]]
    for prev, item in zip([None] + items[:-1], items):
        if prev is not None and math.dist(item[1], prev[2]) > SUBPATH_TOL:
            rings.append([tuple(item[1])])
        rings[-1].append(tuple(item[2]))
    parts = []
    for ring in rings:
        if len(ring) >= 3:
            valid = shapely.make_valid(Polygon(ring))
            flat = shapely.get_parts(shapely.get_parts(valid))  # make_valid nests one level deep
            parts += [g for g in flat if g.geom_type == "Polygon"]
    total = sum(g.area for g in parts)
    kept = [g for g in parts if g.area > SLIVER_MAX_AREA_FRAC * total]
    return shapely.union_all(kept)


def path_line(path: dict) -> LineString | MultiLineString:
    """Centerline of a get_drawings() path of l and c items (page space, pt); c items are sampled
    with bezier_points. Raises ValueError for other items."""
    parts = []
    for item in path["items"]:
        if item[0] == "l":
            parts.append(LineString([tuple(item[1]), tuple(item[2])]))
        elif item[0] == "c":
            parts.append(LineString(bezier_points(np.array([tuple(p) for p in item[1:]]))))
        else:
            raise ValueError("unhandled drawing", item)
    return MultiLineString(parts) if len(parts) > 1 else parts[0]


def circle_of(path: dict) -> tuple[np.ndarray, float] | None:
    """(center, radius) of a get_drawings() path drawn as a circle (page space, pt), else None.

    A circle is made only of c items, ends within JOIN_TOL of where it starts, and every sampled
    point lies within CALLOUT_ROUND_TOL of the radius from the center of their bounding box.
    """
    items = path["items"]
    if not items or any(item[0] != "c" for item in items):
        return None
    if math.dist(items[0][1], items[-1][4]) > JOIN_TOL:
        return None
    pts = np.concatenate([bezier_points(np.array([tuple(p) for p in item[1:]])) for item in items])
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    dist = np.linalg.norm(pts - center, axis=1)
    radius = float(dist.mean())
    if np.abs(dist - radius).max() > CALLOUT_ROUND_TOL:
        return None
    return center, radius


def union_rect(rects: list) -> list[float]:
    """Bounding [x0, y0, x1, y1] of (x0, y0, x1, y1) rects, rounded to 3 decimals (pt)."""
    arr = np.array([tuple(r) for r in rects], dtype=float)
    out = [arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()]
    return [round(float(v), 3) for v in out]


def room_label_json(group: dict) -> dict:
    """A room label group with its rect as 4 [x, y] corners and its dot as [x, y] (pt, rounded to
    3 decimals), for JSON."""
    corners = shapely.get_coordinates(group["rect"].exterior)[:-1]
    dot = group["dot"]
    return {
        "text": group["text"],
        "name": group["name"],
        "rect": [[round(float(x), 3), round(float(y), 3)] for x, y in corners],
        "leader_rect": group["leader_rect"],
        "dot": None if dot is None else [round(dot.x, 3), round(dot.y, 3)],
        "items": group["items"],
    }


def oval_tag_json(group: dict) -> dict:
    """An oval tag group with its rect as 4 [x, y] corners (pt, rounded to 3 decimals), for JSON."""
    corners = shapely.get_coordinates(group["rect"].exterior)[:-1]
    return {
        "text": group["text"],
        "rect": [[round(float(x), 3), round(float(y), 3)] for x, y in corners],
        "leader_rect": group["leader_rect"],
        "items": group["items"],
    }


def words_from_blocks(blocks: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per text line of get_text("dict") blocks: bbox polygon, text, and angle in whole degrees.

    The angle is atan2 of the line's dir in page space (y down), so text read upward is negative.
    Rotated lines get their quad as bbox, axis-aligned lines their rect.
    """
    word_bboxes = []
    word_texts = []
    angles = []

    for block in blocks:
        for line in block.get("lines", []):
            cos, sin = line["dir"]
            angles.append(round(math.degrees(math.atan2(sin, cos))))
            if cos == 0 or sin == 0:
                bbox = shapely.geometry.box(*line["bbox"])
            else:
                q = pymupdf.recover_line_quad(line)
                bbox = Polygon([q.ul, q.ur, q.lr, q.ll])
            text = "".join(span["text"] for span in line["spans"])
            word_bboxes.append(bbox)
            word_texts.append(text)

    return np.array(word_bboxes), np.array(word_texts), np.array(angles)


class FloorPlanParser:
    def __init__(self, page) -> None:
        self.page = page
        self.paths = page.get_drawings()
        self.blocks = page.get_text("dict")["blocks"]
        self.word_bboxes, self.word_texts, self.word_angles = words_from_blocks(self.blocks)

    def parse(self):
        lines = defaultdict(list)
        for path in tqdm(self.paths):
            key = style_key(path)
            if key is None:
                continue

            for item in path["items"]:
                kind = item[0]
                if kind == "l":
                    lines[key].append(LineString([item[1], item[-1]]))
                elif kind not in ("re", "qu", "c"):
                    raise ValueError("unhandled drawing", item)

        # TODO@CLaude: These are only valid regions for one of the files, 
        # To build a robust pipeline, you will need to pre-process files where the drawing area and the 
        # grid lines do not follow this exactly
        self.drawing_area = polygonize(lines[((0.0, 0.0, 0.0), 1.68)])[0] 
        self.grid_key = ((0.498, 0.498, 0.498), 0.24)
        self.grid_lines = lines[self.grid_key]

    def _index_segments(self) -> None:
        """One straight segment per l item (and per c item, as its chord) of every stroked path.

        Sets seg_geoms, seg_path (index into self.paths), seg_width (stroke width, pt), seg_kind
        ("l" or "c"), seg_xy (n, 2, 2), seg_angle (degrees, page space), seg_len (pt), seg_tree,
        and path_segs (path index -> its segment indices).
        """
        geoms, seg_path, seg_width, seg_kind, n_items = [], [], [], [], []
        for i, path in enumerate(self.paths):
            n_items.append(len(path["items"]))
            if "s" not in path["type"]:
                continue
            for item in path["items"]:
                kind = item[0]
                if kind == "l":
                    pts = (item[1], item[2])
                elif kind == "c":
                    pts = (item[1], item[4])
                elif kind in ("re", "qu"):
                    continue  # never part of a measurement; keeps its path out of every group
                else:
                    raise ValueError("unhandled drawing", item)
                geoms.append(LineString([tuple(pts[0]), tuple(pts[1])]))
                seg_path.append(i)
                seg_width.append(path["width"])
                seg_kind.append(kind)
        self.seg_geoms = np.array(geoms)
        self.seg_path = np.array(seg_path, dtype=int)
        self.seg_width = np.array(seg_width, dtype=float)
        self.seg_kind = np.array(seg_kind)
        self.seg_xy = shapely.get_coordinates(self.seg_geoms).reshape(-1, 2, 2)
        d = self.seg_xy[:, 1] - self.seg_xy[:, 0]
        self.seg_angle = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.seg_tree = STRtree(self.seg_geoms)
        self.path_n_items = np.array(n_items)
        self.path_segs = defaultdict(list)
        for k, i in enumerate(self.seg_path):
            self.path_segs[int(i)].append(k)

    def _index_masks(self) -> None:
        """Polygons of fill-only paths made of one re/qu item or only l items (text mask candidates)."""
        polys, mask_path = [], []
        for i, path in enumerate(self.paths):
            if path["type"] != "f":
                continue
            items = path["items"]
            kinds = {item[0] for item in items}
            if not kinds <= {"l", "c", "re", "qu"}:
                raise ValueError("unhandled drawing", items)
            if len(items) == 1 and kinds <= {"re", "qu"}:
                q = items[0][1].quad if items[0][0] == "re" else items[0][1]
                poly = Polygon([q.ul, q.ur, q.lr, q.ll])
            elif kinds == {"l"}:
                poly = Polygon([tuple(item[1]) for item in items] + [tuple(items[-1][2])])
            else:
                continue  # curved or mixed fills are not text masks
            polys.append(poly)
            mask_path.append(i)
        self.mask_polys = np.array(polys)
        self.mask_path = np.array(mask_path, dtype=int)
        self.mask_tree = STRtree(self.mask_polys)

    def _dimension_lines(self, cand: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Candidates that are dimension lines: a tick within OVERSHOOT_MAX of each end.

        A tick is a segment at 45 deg to the line (within ANGLE_TOL), at most TICK_MAX_LEN long,
        with an end within TICK_END_TOL of the line (a slash is two halves starting on the line).
        Its crossing is that end projected onto the line. Returns {segment: (crossings at the two
        ends (2, 2), their positions along the line (2,), end tick segments)}.
        """
        cand = np.unique(np.asarray(cand, dtype=int))
        if len(cand) == 0:
            return {}
        ci, ti = self.seg_tree.query(self.seg_geoms[cand], predicate="dwithin", distance=TICK_END_TOL)
        line = cand[ci]
        ok = (
            (ti != line)
            & (np.abs(angle_diff(self.seg_angle[ti], self.seg_angle[line]) - 45) <= ANGLE_TOL)
            & (self.seg_len[ti] <= TICK_MAX_LEN)
        )
        line, ti = line[ok], ti[ok]
        tick_ends = shapely.points(self.seg_xy[ti])  # (m, 2)
        gap = shapely.distance(tick_ends, self.seg_geoms[line][:, None])
        near_end = gap.argmin(axis=1)
        ok = gap.min(axis=1) <= TICK_END_TOL
        line, ti, near_end = line[ok], ti[ok], near_end[ok]
        pos = shapely.line_locate_point(self.seg_geoms[line], tick_ends[ok, near_end])
        cross_xy = shapely.get_coordinates(shapely.line_interpolate_point(self.seg_geoms[line], pos))
        at_start = pos <= OVERSHOOT_MAX
        at_end = self.seg_len[line] - pos <= OVERSHOOT_MAX
        found = {}
        for j in np.unique(line):
            m = line == j
            s, e = m & at_start, m & at_end
            if not (s.any() and e.any()):
                continue
            ends = np.array([cross_xy[s].mean(axis=0), cross_xy[e].mean(axis=0)])
            found[int(j)] = (ends, np.array([pos[s].mean(), pos[e].mean()]), np.unique(ti[s | e]))
        return found

    def _extension_lines(self, j: int, ends: np.ndarray) -> np.ndarray:
        """Extension line pieces at the end crossings `ends` of dimension line j.

        A piece is perpendicular to j (within ANGLE_TOL), passes within EXT_TOL of a crossing, and
        either crosses j and runs EXT_OVERSHOOT_MIN..OVERSHOOT_MAX past it, or ends at the crossing
        and is that short (the overshoot of a line split at the crossing), or ends at the crossing
        opposite such an overshoot piece. Pieces whose far end joins a line that is neither
        collinear nor a tick are wall edges and are left out.
        """
        ei, si = self.seg_tree.query(shapely.points(ends), predicate="dwithin", distance=EXT_TOL)
        ok = (si != j) & (np.abs(angle_diff(self.seg_angle[si], self.seg_angle[j]) - 90) <= ANGLE_TOL)
        ei, si = ei[ok], si[ok]
        d = self.seg_xy[j, 1] - self.seg_xy[j, 0]
        normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        off = (self.seg_xy[si] - ends[ei][:, None]) @ normal  # (m, 2): end offsets from line j, pt
        near, far = np.abs(off).min(axis=1), np.abs(off).max(axis=1)
        crosses = (off.min(axis=1) < -EXT_TOL) & (off.max(axis=1) > EXT_TOL)
        short = (EXT_OVERSHOOT_MIN <= near) & (near <= OVERSHOOT_MAX)
        at_cross = ~crosses & (near <= EXT_TOL)
        overshoot = at_cross & (EXT_OVERSHOOT_MIN <= far) & (far <= OVERSHOOT_MAX)
        side = np.sign(off[np.arange(len(off)), np.abs(off).argmax(axis=1)])
        opposite = np.array(
            [((ei == e) & overshoot & (side == -sd)).any() for e, sd in zip(ei, side)], dtype=bool
        )
        keep = (crosses & short) | overshoot | (at_cross & (far > OVERSHOOT_MAX) & opposite)
        ei, si, off = ei[keep], si[keep], off[keep]
        far_xy = self.seg_xy[si, np.abs(off).argmax(axis=1)]
        fi, ni = self.seg_tree.query(shapely.points(far_xy), predicate="dwithin", distance=JOIN_TOL)
        gap = np.linalg.norm(self.seg_xy[ni] - far_xy[fi][:, None], axis=2).min(axis=1)
        joined = (
            (ni != si[fi])
            & (gap <= JOIN_TOL)
            & (angle_diff(self.seg_angle[ni], self.seg_angle[si[fi]]) > ANGLE_TOL)
            & ~((np.abs(angle_diff(self.seg_angle[ni], self.seg_angle[si[fi]]) - 45) <= ANGLE_TOL)
                & (self.seg_len[ni] <= TICK_MAX_LEN))
        )
        wall = np.isin(np.arange(len(si)), fi[joined])
        return np.unique(si[~wall])

    def _leader_end(self, xy: np.ndarray, heading: float, exclude: list[int]) -> tuple[str, int]:
        """What a leader's free end at xy touches: ("dim", line), ("next", segment) or ("fail", -1).

        "dim" when xy lies within JOIN_TOL of exactly one dimension line, more than JOIN_TOL inside
        its end crossings (an extension line ends exactly on a crossing).
        Otherwise "next" when exactly one other piece starts within JOIN_TOL of xy and turns at
        most LEADER_MAX_TURN from heading (degrees, direction the leader arrived in).
        """
        point = shapely.points(xy)
        near = self.seg_tree.query(point, predicate="dwithin", distance=JOIN_TOL)
        near = near[~np.isin(near, exclude)]
        hits = []
        for j, (_, span, _) in self._dimension_lines(near).items():
            pos = shapely.line_locate_point(self.seg_geoms[j], point)
            if span.min() + JOIN_TOL < pos < span.max() - JOIN_TOL:
                hits.append(j)
        if len(hits) == 1:
            return "dim", hits[0]
        if hits:
            return "fail", -1
        nxt = self._next_piece(xy, heading, near)
        if nxt is not None:
            return "next", nxt
        return "fail", -1

    def _next_piece(
        self, xy: np.ndarray, heading: float, near: np.ndarray, max_turn: float = LEADER_MAX_TURN
    ) -> int | None:
        """The one segment of near with an end within JOIN_TOL of xy that turns at most max_turn
        from heading (degrees, direction the leader arrived in); None if not exactly one.
        """
        gaps = np.linalg.norm(self.seg_xy[near] - xy, axis=2)  # (m, 2)
        joined = gaps.min(axis=1) <= JOIN_TOL
        near, gaps = near[joined], gaps[joined]
        other = self.seg_xy[near, gaps.argmax(axis=1)]
        out = np.degrees(np.arctan2(other[:, 1] - xy[1], other[:, 0] - xy[0]))
        turn = np.abs((out - heading + 180) % 360 - 180)
        nxt = near[turn <= max_turn]
        return int(nxt[0]) if len(nxt) == 1 else None

    def _find_leader(self, k: int) -> tuple[list[int], int] | str:
        """Leader from text k to a dimension line: (leader segments, line), or a failure reason.

        Starts from every segment with an endpoint within THRSH of the text bbox boundary and
        follows pieces joined end to end. All chains that reach a dimension line must be parts
        of the longest one (a leader can start next to the text with more than one piece).
        """
        bbox = self.word_bboxes[k]
        boundary = bbox.exterior
        found = []
        for s in self.seg_tree.query(boundary, predicate="dwithin", distance=THRSH):
            ends = self.seg_xy[s]
            if shapely.distance(shapely.points(ends), boundary).min() > THRSH:
                continue
            far = int(np.argmax(shapely.distance(shapely.points(ends), bbox)))
            prev, free = ends[1 - far], ends[far]
            chain = [int(s)]
            for _ in range(LEADER_MAX_SEGS):
                heading = math.degrees(math.atan2(free[1] - prev[1], free[0] - prev[0]))
                what, j = self._leader_end(free, heading, chain)
                if what == "dim":
                    found.append((chain, j))
                    break
                if what == "fail":
                    break
                chain.append(j)
                ends = self.seg_xy[j]
                far = int(np.argmax(np.linalg.norm(ends - free, axis=1)))
                prev, free = free, ends[far]
        if not found:
            return "no leader found"
        chain, j = max(found, key=lambda f: len(f[0]))
        if any(fj != j or not set(fc) <= set(chain) for fc, fj in found):
            return f"{len(found)} possible leaders"
        return chain, j

    def _group_paths(self, segs: np.ndarray) -> tuple[list[int], list[int]]:
        """Paths of segs split into (complete, partial): partial paths have items outside segs."""
        segs = {int(s) for s in segs}
        complete, partial = [], []
        for p in sorted({int(self.seg_path[s]) for s in segs}):
            whole = len(self.path_segs[p]) == self.path_n_items[p] and segs.issuperset(self.path_segs[p])
            (complete if whole else partial).append(p)
        return complete, partial

    def get_measurements(self) -> None:
        """Group each dimension text with its dimension line, ticks, extension lines, leader and mask.

        Text next to its line (parallel, centered, within THRSH) or joined to it by a leader.
        Sets self.measurements, a list of {text, rect, leader_rect, items}, where items are
        sorted indices into self.paths and rects are [x0, y0, x1, y1] in page space (pt);
        and self.unclassified_measurements, a list of {text, bbox, reason}. Uses only
        self.paths and the text, so it does not need parse().
        """
        self._index_segments()
        self._index_masks()
        meas = np.flatnonzero([MEAS_TEXT_RE.fullmatch(t) is not None for t in self.word_texts])
        bboxes = self.word_bboxes[meas]

        # Case A: a parallel dimension line next to the text, with the text centered on it.
        ti, si = self.seg_tree.query(bboxes, predicate="dwithin", distance=THRSH)
        t = shapely.line_locate_point(self.seg_geoms[si], shapely.centroid(bboxes[ti]), normalized=True)
        ok = (angle_diff(self.seg_angle[si], self.word_angles[meas][ti]) <= ANGLE_TOL) & (
            np.abs(t - 0.5) <= CENTER_TOL
        )
        ti, si = ti[ok], si[ok]
        dims = self._dimension_lines(si)
        is_dim = np.isin(si, list(dims))
        ti, si = ti[is_dim], si[is_dim]
        dist = shapely.distance(self.seg_geoms[si], bboxes[ti])

        # Text masks: covering most of the text bbox and not much larger than it.
        mi, pi = self.mask_tree.query(bboxes, predicate="intersects")
        cover = shapely.area(shapely.intersection(self.mask_polys[pi], bboxes[mi])) / shapely.area(bboxes[mi])
        ratio = shapely.area(self.mask_polys[pi]) / shapely.area(bboxes[mi])
        is_mask = (cover >= MASK_MIN_COVER) & (ratio <= MASK_MAX_AREA_RATIO)
        mi, mask_paths = mi[is_mask], self.mask_path[pi[is_mask]]

        self.measurements, self.unclassified_measurements = [], []
        claimed: dict[int, str] = {}
        for n, k in enumerate(meas):
            text, bbox = str(self.word_texts[k]), self.word_bboxes[k]
            unclassified = {"text": text, "bbox": [round(float(v), 3) for v in bbox.bounds]}
            leader: list[int] = []
            cands = si[ti == n]
            if len(cands):
                j = int(cands[np.argmin(dist[ti == n])])
            else:
                found = self._find_leader(int(k))
                if isinstance(found, str):
                    self.unclassified_measurements.append({**unclassified, "reason": found})
                    continue
                leader, j = found
                if j not in dims:
                    dims.update(self._dimension_lines(np.array([j])))
            if j in claimed:
                reason = f"dimension line already used by {claimed[j]!r}"
                self.unclassified_measurements.append({**unclassified, "reason": reason})
                continue
            ends, _, ticks = dims[j]
            body = np.concatenate([[j], ticks, self._extension_lines(j, ends)])
            body_paths, body_partial = self._group_paths(body)
            leader_paths, leader_partial = self._group_paths(np.array(leader, dtype=int))
            if body_partial or leader_partial:
                reason = f"paths {body_partial + leader_partial} have items outside the group"
                self.unclassified_measurements.append({**unclassified, "reason": reason})
                continue
            masks = mask_paths[mi == n]
            if len(masks) > 1:
                reason = f"{len(masks)} candidate masks {masks.tolist()}"
                self.unclassified_measurements.append({**unclassified, "reason": reason})
                continue
            claimed[j] = text
            rect_paths = body_paths + masks.tolist()
            leader_rect = union_rect([self.paths[p]["rect"] for p in leader_paths]) if leader else None
            self.measurements.append(
                {
                    "text": text,
                    "rect": union_rect([bbox.bounds] + [self.paths[p]["rect"] for p in rect_paths]),
                    "leader_rect": leader_rect,
                    "items": sorted(set(rect_paths + leader_paths)),
                }
            )
        n_leader = sum(m["leader_rect"] is not None for m in self.measurements)
        print(
            f"measurements: {len(meas)} texts, {len(self.measurements) - n_leader} next to their line, "
            f"{n_leader} with a leader, {len(self.unclassified_measurements)} unclassified"
        )

    def _tag_bars(self, xy: np.ndarray, prev: np.ndarray, last: int) -> np.ndarray:
        """End bar candidates at a tag leader's free end xy; the last piece (segment last) runs
        from prev to xy.

        A bar has one end within JOIN_TOL of xy and runs back along the leader (it may pass the
        leader's last bend), is parallel to the last piece (ANGLE_TOL), at most TAG_BAR_MAX_LEN
        long, and at least TAG_BAR_MIN_WIDTH_RATIO times as wide.
        """
        near = self.seg_tree.query(shapely.points(xy), predicate="dwithin", distance=JOIN_TOL)
        near = near[near != last]
        gaps = np.linalg.norm(self.seg_xy[near] - xy, axis=2)  # (m, 2)
        other = self.seg_xy[near, gaps.argmax(axis=1)]
        ok = (
            (gaps.min(axis=1) <= JOIN_TOL)
            & ((other - xy) @ (prev - xy) > 0)
            & (angle_diff(self.seg_angle[near], self.seg_angle[last]) <= ANGLE_TOL)
            & (self.seg_len[near] <= TAG_BAR_MAX_LEN)
            & (self.seg_width[near] >= TAG_BAR_MIN_WIDTH_RATIO * self.seg_width[last])
        )
        return near[ok]

    def _find_tag_leaders(
        self, diamond: Polygon, edges: np.ndarray, all_edges: np.ndarray, copies: int
    ) -> list[tuple[list[int], int]] | str:
        """Leaders from a diamond's corners to their end bars: [(leader segments, bar)], or a
        failure reason.

        edges are the diamond's segments, all_edges those of every diamond (tags can touch at
        their corners). A diamond drawn `copies` times has one leader per copy: exactly that many
        other segments with the stroke width of the diamond's edges must start within JOIN_TOL of
        a corner. Pieces of that width are followed with _next_piece (TAG_LEADER_MAX_TURN) until
        a free end has exactly one end bar (_tag_bars).
        """
        corners = np.array(diamond.exterior.coords[:4])
        ci, si = self.seg_tree.query(shapely.points(corners), predicate="dwithin", distance=JOIN_TOL)
        ok = ~np.isin(si, all_edges) & np.isin(self.seg_width[si], self.seg_width[edges])
        ci, si = ci[ok], si[ok]
        gaps = np.linalg.norm(self.seg_xy[si] - corners[ci][:, None], axis=2)  # (m, 2)
        starts = np.flatnonzero(gaps.min(axis=1) <= JOIN_TOL)
        if len(starts) != copies or len(np.unique(si[starts])) != copies:
            return f"{len(np.unique(si[starts]))} leader starts at the corners for {copies} copies"
        found = []
        for s, corner in zip(si[starts].tolist(), corners[ci[starts]]):
            ends = self.seg_xy[s]
            far = int(np.argmax(np.linalg.norm(ends - corner, axis=1)))
            prev, free = ends[1 - far], ends[far]
            chain = [s]
            for _ in range(LEADER_MAX_SEGS):
                bars = self._tag_bars(free, prev, chain[-1])
                bars = bars[~np.isin(bars, chain)]
                if len(bars) == 1:
                    found.append((chain, int(bars[0])))
                    break
                if len(bars) > 1:
                    return f"{len(bars)} end bars"
                heading = math.degrees(math.atan2(free[1] - prev[1], free[0] - prev[0]))
                near = self.seg_tree.query(shapely.points(free), predicate="dwithin", distance=JOIN_TOL)
                near = near[~np.isin(near, np.concatenate([chain, all_edges]))]
                near = near[self.seg_width[near] == self.seg_width[s]]
                j = self._next_piece(free, heading, near, TAG_LEADER_MAX_TURN)
                if j is None:
                    return "leader ends without an end bar"
                chain.append(j)
                ends = self.seg_xy[j]
                far = int(np.argmax(np.linalg.norm(ends - free, axis=1)))
                prev, free = free, ends[far]
            else:
                return f"no end bar within {LEADER_MAX_SEGS} leader pieces"
        return found

    def get_tags(self) -> None:
        """Group each tag: diamond, the code inside it, its mask, leader and end bar.

        A diamond is a square of straight stroked segments at 45 deg to the axes (within
        ANGLE_TOL) around the center of one text line, or of identical text lines drawn on top
        of each other (a tag drawn several times, one leader per copy). It has one mask per copy:
        a fill whose bounds match the diamond's bbox within TAG_MASK_TOL. Leaders run from its
        corners to end bars (_find_tag_leaders). Sets self.tags, a list of {text, rect, items}
        (items: sorted indices into self.paths; rect: [x0, y0, x1, y1], page space, pt), and
        self.unclassified_tags, a list of {text, bbox (the diamond's), reason}. Uses only
        self.paths and the text, so it does not need parse().
        """
        self._index_segments()
        self._index_masks()
        diag = np.flatnonzero(
            (np.abs(angle_diff(self.seg_angle, 0) - 45) <= ANGLE_TOL) & (self.seg_kind == "l")
        )
        rects, _ = find_rectangles(self.seg_geoms[diag])
        rects = np.array(rects, dtype=object)
        sides = np.array([np.linalg.norm(np.diff(r.exterior.coords[:3], axis=0), axis=1) for r in rects])
        squares = rects[np.abs(sides[:, 0] - sides[:, 1]) <= JOIN_TOL] if len(rects) else rects
        sq, wi = STRtree(shapely.centroid(self.word_bboxes)).query(squares, predicate="contains")
        texts_in = defaultdict(list)  # square -> text lines with their center inside it
        for q, k in zip(sq.tolist(), wi.tolist()):
            texts_in[q].append(k)
        by_text = defaultdict(list)  # first of identical stacked text lines -> squares around them
        for q, ks in texts_in.items():
            b = shapely.bounds(self.word_bboxes[ks])
            if len(set(self.word_texts[ks])) == 1 and np.abs(b - b[0]).max() <= TAG_EDGE_TOL:
                by_text[min(ks)].append(q)
        diamond_edges = {}  # square -> its segments
        for q in sorted({q for qs in by_text.values() for q in qs}):
            edges = self.seg_tree.query(squares[q].exterior.buffer(TAG_EDGE_TOL), predicate="contains")
            diamond_edges[q] = edges[np.isin(edges, diag)]
        all_edges = np.concatenate([np.zeros(0, dtype=int), *diamond_edges.values()])

        self.tags, self.unclassified_tags = [], []
        claimed: dict[int, str] = {}  # path index -> text of the tag that owns it
        n_pieces = defaultdict(int)  # leader pieces -> leaders
        for k in sorted(by_text):
            text = str(self.word_texts[k])
            if len(by_text[k]) > 1:
                bbox = union_rect([squares[q].bounds for q in by_text[k]])
                reason = f"{len(by_text[k])} diamonds around the text"
                self.unclassified_tags.append({"text": text, "bbox": bbox, "reason": reason})
                continue
            q = by_text[k][0]
            diamond, edges, copies = squares[q], diamond_edges[q], len(texts_in[q])
            unclassified = {"text": text, "bbox": [round(float(v), 3) for v in diamond.bounds]}
            mask_cand = self.mask_tree.query(diamond, predicate="intersects")
            off = np.abs(shapely.bounds(self.mask_polys[mask_cand]) - np.array(diamond.bounds))
            masks = self.mask_path[mask_cand[off.max(axis=1) <= TAG_MASK_TOL]] if len(mask_cand) else []
            if len(masks) != copies:
                reason = f"{len(masks)} candidate masks for {copies} copies"
                self.unclassified_tags.append({**unclassified, "reason": reason})
                continue
            found = self._find_tag_leaders(diamond, edges, all_edges, copies)
            if isinstance(found, str):
                self.unclassified_tags.append({**unclassified, "reason": found})
                continue
            segs = np.concatenate([edges] + [chain + [bar] for chain, bar in found])
            paths, partial = self._group_paths(segs)
            if partial:
                reason = f"paths {partial} have items outside the group"
                self.unclassified_tags.append({**unclassified, "reason": reason})
                continue
            items = sorted(set(paths + masks.tolist()))
            taken = sorted({claimed[p] for p in items if p in claimed})
            if taken:
                reason = f"paths already used by tags {taken}"
                self.unclassified_tags.append({**unclassified, "reason": reason})
                continue
            claimed.update({p: text for p in items})
            for chain, _ in found:
                n_pieces[len(chain)] += 1
            rect = union_rect([self.word_bboxes[k].bounds] + [self.paths[p]["rect"] for p in items])
            self.tags.append({"text": text, "rect": rect, "items": items})
        widest = self.seg_width.max()
        wide_paths = set(self.seg_path[np.isclose(self.seg_width, widest)].tolist())
        print(
            f"tags: {len(by_text)} diamonds with a text, {len(self.tags)} grouped "
            f"(leader pieces -> leaders: {dict(sorted(n_pieces.items()))}), "
            f"{len(self.unclassified_tags)} unclassified; widest strokes ({widest:.4g} pt): "
            f"{len(wide_paths - set(claimed))} of {len(wide_paths)} paths not in a tag"
        )

    def _clipped_fills(self) -> list[tuple[dict, int]]:
        """(clip, index into self.paths) of every fill-only path clipped on its own.

        In get_drawings(extended=True) a level-0 clip followed directly by one level-1 fill-only
        path clips that path; clips of a stroke or of more than one entry are left out.
        """
        ext = self.page.get_drawings(extended=True)
        painted = [p["seqno"] for p in ext if p["type"] != "clip"]
        if painted != [p["seqno"] for p in self.paths]:
            raise RuntimeError("extended drawings do not match get_drawings() by seqno")
        seq_to_path = {p["seqno"]: i for i, p in enumerate(self.paths)}
        found = []
        for k, clip in enumerate(ext[:-1]):
            if clip["type"] != "clip" or clip["level"] != 0:
                continue
            clipped, after = ext[k + 1], ext[k + 2] if k + 2 < len(ext) else None
            if clipped["type"] != "f" or clipped["level"] != 1 or (after is not None and after["level"] >= 1):
                continue
            found.append((clip, seq_to_path[clipped["seqno"]]))
        return found

    def _index_arrowheads(self) -> None:
        """Clipped fills small enough to be arrowheads, represented by their clip area.

        Candidates are clipped fills (_clipped_fills) whose clip is only l items, whose scissor
        fits in ALIGN_HEAD_MAX_LEN and whose area is one polygon (clip_polygon). Sets head_polys,
        head_path (index into self.paths of the fill), head_color (fill color), head_len and
        head_width (long and short side of the min rotated rectangle, pt), and head_tree.
        """
        polys, head_path, head_color = [], [], []
        for clip, i in self._clipped_fills():
            scissor = clip["scissor"]
            if {item[0] for item in clip["items"]} != {"l"} or max(scissor.width, scissor.height) > ALIGN_HEAD_MAX_LEN:
                continue
            area = clip_polygon(clip["items"])
            if not isinstance(area, Polygon):
                continue  # several separate shapes
            polys.append(area)
            head_path.append(i)
            head_color.append(self.paths[i]["fill"])
        self.head_polys = np.array(polys)
        self.head_path = np.array(head_path, dtype=int)
        self.head_color = head_color
        rects = shapely.minimum_rotated_rectangle(self.head_polys)
        corners = np.array([np.array(r.exterior.coords)[:3] for r in rects]).reshape(-1, 3, 2)
        sides = np.diff(corners, axis=1)  # (n, 2, 2): two adjacent sides
        lens = np.linalg.norm(sides, axis=2)
        self.head_len = lens.max(axis=1)
        self.head_width = lens.min(axis=1)
        self.head_tree = STRtree(self.head_polys)

    def _align_arrowhead(self, xy: np.ndarray, out: np.ndarray, leg: int, color: tuple) -> int | str:
        """The one arrowhead at a leg's far end xy, pointing along unit vector out (away from
        the crossbar): index into head_polys, or a failure reason.

        It lies within JOIN_TOL of xy, has fill color `color`, its min rotated rectangle is at
        most ALIGN_HEAD_MAX_LEN long and ALIGN_HEAD_MIN_ASPECT times as long as wide, and its
        tip (the vertex farthest from xy) lies beyond xy in the leg's direction (ANGLE_TOL).
        (The min rotated rectangle of a thin triangle follows one of its slanted sides, so its
        direction is not the arrowhead's axis.)
        """
        cand = self.head_tree.query(shapely.points(xy), predicate="dwithin", distance=JOIN_TOL)
        tips = np.zeros((len(cand), 2))
        for n, h in enumerate(cand):
            pts = shapely.get_coordinates(self.head_polys[h].exterior)
            tips[n] = pts[np.linalg.norm(pts - xy, axis=1).argmax()]
        to_tip = tips - xy
        tip_angle = np.degrees(np.arctan2(to_tip[:, 1], to_tip[:, 0]))
        ok = (
            (to_tip @ out > 0)
            & (angle_diff(tip_angle, self.seg_angle[leg]) <= ANGLE_TOL)
            & (self.head_len[cand] <= ALIGN_HEAD_MAX_LEN)
            & (self.head_len[cand] >= ALIGN_HEAD_MIN_ASPECT * self.head_width[cand])
            & np.array([np.allclose(self.head_color[h], color) for h in cand], dtype=bool)
        )
        if ok.sum() != 1:
            return f"{ok.sum()} arrowheads at a leg end"
        return int(cand[ok][0])

    def _align_bracket(self, bar: int, center: np.ndarray) -> tuple[list[int], list[int]] | str:
        """Legs and arrowheads of the ALIGN bracket with crossbar segment bar, whose text is
        centered at center: (segments [bar, leg, leg], arrowheads [h, h]), or a failure reason.

        At each crossbar end exactly one leg: a segment with an end within JOIN_TOL, perpendicular
        (ANGLE_TOL), at most ALIGN_LEG_MAX_LEN long, with the crossbar's stroke width and color,
        running away from the text. Each leg's far end has one arrowhead (_align_arrowhead).
        """
        path = self.paths[self.seg_path[bar]]
        xy = self.seg_xy[bar]
        d = xy[1] - xy[0]
        normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        away = -np.sign((center - xy[0]) @ normal) * normal  # from the crossbar, away from the text
        legs, heads = [], []
        for end in xy:
            near = self.seg_tree.query(shapely.points(end), predicate="dwithin", distance=JOIN_TOL)
            near = near[(near != bar) & (self.seg_kind[near] == "l")]
            gaps = np.linalg.norm(self.seg_xy[near] - end, axis=2)  # (m, 2)
            far = self.seg_xy[near, gaps.argmax(axis=1)]
            ok = (
                (gaps.min(axis=1) <= JOIN_TOL)
                & (np.abs(angle_diff(self.seg_angle[near], self.seg_angle[bar]) - 90) <= ANGLE_TOL)
                & (self.seg_len[near] <= ALIGN_LEG_MAX_LEN)
                & np.isclose(self.seg_width[near], path["width"])
                & np.array(
                    [np.allclose(self.paths[p]["color"], path["color"]) for p in self.seg_path[near]], dtype=bool
                )
                & ((far - end) @ away > 0)
            )
            if ok.sum() != 1:
                return f"{ok.sum()} legs at a crossbar end"
            leg, tip = int(near[ok][0]), far[ok][0]
            head = self._align_arrowhead(tip, away, leg, path["color"])
            if isinstance(head, str):
                return head
            legs.append(leg)
            heads.append(head)
        if heads[0] == heads[1]:
            return "both legs end on the same arrowhead"
        return [bar] + legs, heads

    def get_align_markers(self) -> None:
        """Group each ALIGN marker: the text ALIGN_TEXT, its bracket and both arrowheads.

        The crossbar is a straight segment parallel to the text (ANGLE_TOL), within
        ALIGN_TEXT_MAX_GAP of its bbox, with the text center projecting inside it. Legs and
        arrowheads are found by _align_bracket; exactly one crossbar candidate must have them.
        There is no text mask. Sets self.align_markers, a list of {text, rect, items} (items:
        sorted indices into self.paths; rect: [x0, y0, x1, y1], page space, pt), and
        self.unclassified_align_markers, a list of {text, bbox (the text's), reason}. Uses only
        self.paths and the text, so it does not need parse().
        """
        self._index_segments()
        self._index_arrowheads()
        self.align_markers, self.unclassified_align_markers = [], []
        claimed: dict[int, str] = {}  # path index -> bbox of the marker text that owns it
        used_heads: set[int] = set()
        texts = np.flatnonzero(self.word_texts == ALIGN_TEXT)
        for k in texts:
            text, bbox = str(self.word_texts[k]), self.word_bboxes[k]
            unclassified = {"text": text, "bbox": [round(float(v), 3) for v in bbox.bounds]}
            center = shapely.get_coordinates(shapely.centroid(bbox))[0]
            cand = self.seg_tree.query(bbox, predicate="dwithin", distance=ALIGN_TEXT_MAX_GAP)
            cand = cand[
                (self.seg_kind[cand] == "l") & (angle_diff(self.seg_angle[cand], self.word_angles[k]) <= ANGLE_TOL)
            ]
            t = shapely.line_locate_point(self.seg_geoms[cand], shapely.points(center), normalized=True)
            cand = cand[(t > 0) & (t < 1)]
            found = [b for b in (self._align_bracket(int(c), center) for c in cand) if not isinstance(b, str)]
            if len(found) != 1:
                reason = "no bracket" if not found else f"{len(found)} brackets"
                self.unclassified_align_markers.append({**unclassified, "reason": reason})
                continue
            segs, heads = found[0]
            paths, partial = self._group_paths(np.array(segs))
            if partial:
                reason = f"paths {partial} have items outside the group"
                self.unclassified_align_markers.append({**unclassified, "reason": reason})
                continue
            items = sorted(set(paths + self.head_path[heads].tolist()))
            taken = sorted({claimed[p] for p in items if p in claimed})
            if taken:
                reason = f"paths already used by markers at {taken}"
                self.unclassified_align_markers.append({**unclassified, "reason": reason})
                continue
            claimed.update({p: str(unclassified["bbox"]) for p in items})
            used_heads.update(heads)
            rect = union_rect([bbox.bounds] + [self.paths[p]["rect"] for p in items])
            self.align_markers.append({"text": text, "rect": rect, "items": items})
        shaped = np.flatnonzero(
            (self.head_len <= ALIGN_HEAD_MAX_LEN) & (self.head_len >= ALIGN_HEAD_MIN_ASPECT * self.head_width)
        )
        print(
            f"align markers: {len(texts)} {ALIGN_TEXT!r} texts, {len(self.align_markers)} grouped, "
            f"{len(self.unclassified_align_markers)} unclassified; "
            f"{len(set(shaped.tolist()) - used_heads)} of {len(shaped)} arrowhead-shaped clipped fills not in a marker"
        )

    def get_callouts(self) -> None:
        """Group each reference callout: circle, code, black parts, their outlines, view numbers
        and masks.

        A callout circle is a stroked path that circle_of accepts, with the center of exactly one
        text line (the code) inside it. Its black parts are clipped fills (_clipped_fills) of the
        circle's stroke color whose clip (only l items, one polygon) touches the circle within
        CALLOUT_TOUCH_TOL from outside and fits in its diameter; a circle without one is not a
        callout. Outlines are stroked paths of the circle's width and color lying entirely within
        CALLOUT_TOUCH_TOL of the circle or a part's clip boundary. Each part has exactly one view
        number (CALLOUT_NUMBER_RE) within CALLOUT_NUMBER_MAX_GAP. Text masks are found as in
        get_measurements, at most one per text. Sets self.callouts, a list of {text, rect, items}
        (items: sorted indices into self.paths; rect: [x0, y0, x1, y1], page space, pt), and
        self.unclassified_callouts, a list of {text, bbox (the circle's), reason}. Uses only
        self.paths and the text, so it does not need parse().
        """
        self._index_masks()
        circ_path, centers, radii = [], [], []
        for i, path in enumerate(self.paths):
            found = circle_of(path) if "s" in path["type"] else None
            if found is not None:
                circ_path.append(i)
                centers.append(found[0])
                radii.append(found[1])
        circ_path = np.array(circ_path, dtype=int)
        disks = shapely.buffer(shapely.points(np.array(centers).reshape(-1, 2)), np.array(radii), quad_segs=64)
        ci, wi = STRtree(shapely.centroid(self.word_bboxes)).query(disks, predicate="contains")
        texts_in = defaultdict(list)  # circle -> text lines with their center inside it
        for c, k in zip(ci.tolist(), wi.tolist()):
            texts_in[c].append(k)
        coded = sorted(c for c, ks in texts_in.items() if len(ks) == 1)

        # Black part candidates: clipped fills whose clip is one polygon of l items.
        part_polys, part_path = [], []
        for clip, i in self._clipped_fills():
            if {item[0] for item in clip["items"]} != {"l"} or self.paths[i]["fill"] is None:
                continue
            area = clip_polygon(clip["items"])
            if isinstance(area, Polygon):
                part_polys.append(area)
                part_path.append(i)
        part_polys, part_path = np.array(part_polys, dtype=object), np.array(part_path, dtype=int)
        rings = np.array([path_line(self.paths[circ_path[c]]) for c in coded], dtype=object)
        ri, pi = STRtree(part_polys).query(rings, predicate="dwithin", distance=CALLOUT_TOUCH_TOL)
        size = np.ptp(shapely.bounds(part_polys[pi]).reshape(-1, 2, 2), axis=1).max(axis=1)
        inner = shapely.buffer(disks[np.array(coded, dtype=int)[ri]], -CALLOUT_TOUCH_TOL)
        ok = (
            np.array(
                [np.allclose(self.paths[part_path[p]]["fill"], self.paths[circ_path[coded[r]]]["color"]) for r, p in zip(ri, pi)],
                dtype=bool,
            )
            & (size <= 2 * np.array(radii)[np.array(coded, dtype=int)[ri]])
            & ~shapely.intersects(part_polys[pi], inner)
        )
        ri, pi = ri[ok], pi[ok]

        # Outline candidates: stroked paths of only l and c items.
        line_path = np.array(
            [i for i, p in enumerate(self.paths) if "s" in p["type"] and p["items"] and {it[0] for it in p["items"]} <= {"l", "c"}],
            dtype=int,
        )
        line_tree = STRtree(np.array([path_line(self.paths[i]) for i in line_path], dtype=object))
        num_tree = STRtree(self.word_bboxes)

        self.callouts, self.unclassified_callouts = [], []
        claimed: dict[int, str] = {}  # path index -> bbox of the circle that owns it
        used_parts: set[int] = set()
        n_bare = 0
        for r, c in enumerate(coded):
            circle = self.paths[circ_path[c]]
            k = texts_in[c][0]
            text = str(self.word_texts[k])
            unclassified = {"text": text, "bbox": [round(float(v), 3) for v in circle["rect"]]}
            parts = pi[ri == r]
            if len(parts) == 0:
                n_bare += 1
                continue
            zone = shapely.buffer(
                shapely.union_all([rings[r]] + [p.exterior for p in part_polys[parts]]), CALLOUT_TOUCH_TOL
            )
            lines = line_path[line_tree.query(zone, predicate="contains")]
            lines = [
                int(i)
                for i in lines
                if np.isclose(self.paths[i]["width"], circle["width"])
                and np.allclose(self.paths[i]["color"], circle["color"])
            ]
            numbers, reason = [], None
            for p in parts:
                near = num_tree.query(part_polys[p], predicate="dwithin", distance=CALLOUT_NUMBER_MAX_GAP)
                near = [int(n) for n in near if CALLOUT_NUMBER_RE.fullmatch(str(self.word_texts[n]))]
                if len(near) != 1:
                    reason = f"{len(near)} view numbers at a black part"
                    break
                numbers += near
            if reason is None and len(set(numbers)) != len(numbers):
                reason = "black parts share a view number"
            masks = []
            if reason is None:
                for t in [k] + numbers:
                    bbox = self.word_bboxes[t]
                    cand = self.mask_tree.query(bbox, predicate="intersects")
                    cover = shapely.area(shapely.intersection(self.mask_polys[cand], bbox)) / bbox.area
                    ratio = shapely.area(self.mask_polys[cand]) / bbox.area
                    found = self.mask_path[cand[(cover >= MASK_MIN_COVER) & (ratio <= MASK_MAX_AREA_RATIO)]]
                    if len(found) > 1:
                        reason = f"{len(found)} candidate masks {found.tolist()} for {self.word_texts[t]!r}"
                        break
                    masks += found.tolist()
            if reason is not None:
                self.unclassified_callouts.append({**unclassified, "reason": reason})
                continue
            items = sorted({int(circ_path[c]), *part_path[parts].tolist(), *lines, *masks})
            taken = sorted({claimed[p] for p in items if p in claimed})
            if taken:
                reason = f"paths already used by callouts at {taken}"
                self.unclassified_callouts.append({**unclassified, "reason": reason})
                continue
            claimed.update({p: str(unclassified["bbox"]) for p in items})
            used_parts.update(parts.tolist())
            boxes = [self.word_bboxes[t].bounds for t in [k] + numbers]
            rect = union_rect(boxes + [self.paths[p]["rect"] for p in items])
            self.callouts.append({"text": text, "rect": rect, "items": items})
        print(
            f"callouts: {len(coded)} circles around one text, {len(self.callouts)} grouped, "
            f"{len(self.unclassified_callouts)} unclassified, {n_bare} without a black part; "
            f"{len(set(pi.tolist()) - used_parts)} of {len(set(pi.tolist()))} black parts touching "
            f"a circle not in a callout"
        )

    def _label_leader(
        self,
        region: Polygon | MultiPolygon,
        exclude: np.ndarray,
        target_center: np.ndarray,
        target_radius: np.ndarray,
        max_turn: float,
    ) -> tuple[list[int], int] | str | None:
        """Leader from a label (region: a room label's box and name lines, or an oval tag's oval) to
        a target (an end dot or an arrowhead tip): (leader segments, target), None if no chain
        reaches a target, or a failure reason.

        Starts from every segment not in exclude with an end within THRSH of region and follows
        pieces joined end to end (_next_piece, max_turn degrees) until the free end lies within a
        target's radius plus JOIN_TOL. Chains ending anywhere else (walls, other annotations) are
        ignored. All chains that reach a target must reach the same one and be parts of the longest.
        """
        found = []
        for s in self.seg_tree.query(region, predicate="dwithin", distance=THRSH):
            if s in exclude:
                continue
            ends = self.seg_xy[s]
            dist = shapely.distance(shapely.points(ends), region)
            if dist.min() > THRSH:
                continue
            far = int(np.argmax(dist))
            prev, free = ends[1 - far], ends[far]
            chain = [int(s)]
            for _ in range(LEADER_MAX_SEGS):
                hit = np.flatnonzero(np.linalg.norm(target_center - free, axis=1) <= target_radius + JOIN_TOL)
                if len(hit) > 1:
                    return f"leader ends on {len(hit)} targets"
                if len(hit) == 1:
                    found.append((chain, int(hit[0])))
                    break
                heading = math.degrees(math.atan2(free[1] - prev[1], free[0] - prev[0]))
                near = self.seg_tree.query(shapely.points(free), predicate="dwithin", distance=JOIN_TOL)
                near = near[~np.isin(near, np.concatenate([chain, exclude]))]
                j = self._next_piece(free, heading, near, max_turn)
                if j is None:
                    break
                chain.append(j)
                ends = self.seg_xy[j]
                far = int(np.argmax(np.linalg.norm(ends - free, axis=1)))
                prev, free = free, ends[far]
        if not found:
            return None
        chain, target = max(found, key=lambda f: len(f[0]))
        if any(ft != target or not set(fc) <= set(chain) for fc, ft in found):
            return f"{len(found)} possible leaders"
        return chain, target

    def get_room_labels(self) -> None:
        """Group each room label: number, name lines, number box, text masks, leader and end dot.

        A room number is a horizontal text line matching ROOM_NUMBER_RE inside a box: a rectangle
        of straight stroked segments (find_rectangles on the segments within THRSH of the text)
        that contains the text center and is at most ROOM_BOX_MAX_AREA_RATIO times the text bbox
        area. Matching text without a box is not a room label. The box edges are all segments
        within ROOM_EDGE_TOL of its outline (each edge is drawn twice). Name lines are stacked
        above the box: each next line is horizontal, its bottom within ROOM_NAME_MAX_GAP above the
        current top and its center within ROOM_NAME_CENTER_TOL of the box center. The number's
        mask is a fill matching the box within ROOM_MASK_TOL; name masks are found as in
        get_measurements; at most one per text. End dots are filled circles (circle_of) of radius
        at most ROOM_DOT_MAX_RADIUS. The leader (_label_leader) is optional; a dot reached from
        more than one label leaves all of them unclassified.

        Sets self.enc_labels, a list of {kind ("room_label"), text, name, rect, leader_rect, dot, items}: rect is the
        minimum rotated rectangle of the box and name lines (Polygon), leader_rect the bounding
        [x0, y0, x1, y1] of the leader or None, dot the end dot's center (Point) or None, and
        items sorted indices into self.paths (page space, pt). Sets self.unclassified_room_labels,
        a list of {text, bbox (the box's, else the text's), reason}. Uses only self.paths and
        the text, so it does not need parse().
        """
        self._index_segments()
        self._index_masks()
        word_bounds = shapely.bounds(self.word_bboxes)
        word_cx = (word_bounds[:, 0] + word_bounds[:, 2]) / 2
        dot_path, dot_center, dot_radius = [], [], []
        for i, path in enumerate(self.paths):
            found = circle_of(path) if "f" in path["type"] else None
            if found is not None and found[1] <= ROOM_DOT_MAX_RADIUS:
                dot_path.append(i)
                dot_center.append(found[0])
                dot_radius.append(found[1])
        dot_center = np.array(dot_center, dtype=float).reshape(-1, 2)
        dot_radius = np.array(dot_radius, dtype=float)

        self.enc_labels, self.unclassified_room_labels = [], []
        numbers = np.flatnonzero([ROOM_NUMBER_RE.fullmatch(t) is not None for t in self.word_texts])
        cands = []  # labels whose parts were all found, before dot ownership and path checks
        n_no_box = 0
        for k in numbers:
            text, bbox = str(self.word_texts[k]), self.word_bboxes[k]
            unclassified = {"text": text, "bbox": [round(float(v), 3) for v in bbox.bounds]}
            near = self.seg_tree.query(bbox, predicate="dwithin", distance=THRSH)
            rects, _ = find_rectangles(self.seg_geoms[near[self.seg_kind[near] == "l"]])
            center = shapely.centroid(bbox)
            boxes = [r for r in rects if r.contains(center) and r.area <= ROOM_BOX_MAX_AREA_RATIO * bbox.area]
            if not boxes:
                n_no_box += 1
                continue
            if len(boxes) > 1:
                self.unclassified_room_labels.append({**unclassified, "reason": f"{len(boxes)} boxes"})
                continue
            box = boxes[0]
            unclassified["bbox"] = [round(float(v), 3) for v in box.bounds]
            if self.word_angles[k] != 0:
                self.unclassified_room_labels.append({**unclassified, "reason": "number not horizontal"})
                continue
            edges = self.seg_tree.query(box.exterior.buffer(ROOM_EDGE_TOL), predicate="contains")

            x0, top, x1, _ = box.bounds
            name: list[int] = []
            reason = None
            while True:
                gap = top - word_bounds[:, 3]
                above = np.flatnonzero(
                    (self.word_angles == 0)
                    & (gap >= 0)
                    & (gap <= ROOM_NAME_MAX_GAP)
                    & (np.abs(word_cx - (x0 + x1) / 2) <= ROOM_NAME_CENTER_TOL)
                )
                if len(above) != 1:
                    if len(above) > 1:
                        reason = f"{len(above)} candidate name lines"
                    break
                name.insert(0, int(above[0]))
                top = word_bounds[above[0], 1]
            if reason is None and not name:
                reason = "no name above the box"

            masks: list[int] = []
            if reason is None:
                cand = self.mask_tree.query(box, predicate="intersects")
                off = np.abs(shapely.bounds(self.mask_polys[cand]) - np.array(box.bounds)).reshape(-1, 4)
                masks = self.mask_path[cand[off.max(axis=1, initial=0.0) <= ROOM_MASK_TOL]].tolist()
                if len(masks) > 1:
                    reason = f"{len(masks)} candidate masks {masks} for {text!r}"
            for t in name if reason is None else []:
                tb = self.word_bboxes[t]
                cand = self.mask_tree.query(tb, predicate="intersects")
                cover = shapely.area(shapely.intersection(self.mask_polys[cand], tb)) / tb.area
                ratio = shapely.area(self.mask_polys[cand]) / tb.area
                found = self.mask_path[cand[(cover >= MASK_MIN_COVER) & (ratio <= MASK_MAX_AREA_RATIO)]]
                if len(found) > 1:
                    reason = f"{len(found)} candidate masks {found.tolist()} for {self.word_texts[t]!r}"
                    break
                masks += found.tolist()

            leader = None
            if reason is None:
                region = shapely.union_all([box, *self.word_bboxes[name]])
                leader = self._label_leader(region, edges, dot_center, dot_radius, ROOM_LEADER_MAX_TURN)
                if isinstance(leader, str):
                    reason = leader
            if reason is not None:
                self.unclassified_room_labels.append({**unclassified, "reason": reason})
                continue
            cands.append((text, unclassified, box, edges, name, masks, leader))

        reached = defaultdict(int)  # dot -> labels whose leader reaches it
        for *_, leader in cands:
            if leader is not None:
                reached[leader[1]] += 1
        claimed: dict[int, str] = {}  # path index -> bbox of the label box that owns it
        used_dots: set[int] = set()
        for text, unclassified, box, edges, name, masks, leader in cands:
            chain, dot = leader if leader is not None else ([], None)
            if dot is not None and reached[dot] > 1:
                reason = f"dot reached from {reached[dot]} labels"
                self.unclassified_room_labels.append({**unclassified, "reason": reason})
                continue
            paths, partial = self._group_paths(np.concatenate([edges, chain]).astype(int))
            if partial:
                reason = f"paths {partial} have items outside the group"
                self.unclassified_room_labels.append({**unclassified, "reason": reason})
                continue
            dot_items = [dot_path[dot]] if dot is not None else []
            items = sorted(set(paths + masks + dot_items))
            taken = sorted({claimed[p] for p in items if p in claimed})
            if taken:
                reason = f"paths already used by labels at {taken}"
                self.unclassified_room_labels.append({**unclassified, "reason": reason})
                continue
            claimed.update({p: str(unclassified["bbox"]) for p in items})
            leader_paths = sorted({int(self.seg_path[s]) for s in chain})
            if dot is not None:
                used_dots.add(dot)
            self.enc_labels.append(
                {
                    "kind": "room_label",
                    "text": text,
                    "name": " ".join(str(self.word_texts[t]) for t in name),
                    "rect": shapely.minimum_rotated_rectangle(shapely.union_all([box, *self.word_bboxes[name]])),
                    "leader_rect": union_rect([self.paths[p]["rect"] for p in leader_paths]) if chain else None,
                    "dot": shapely.Point(dot_center[dot]) if dot is not None else None,
                    "items": items,
                }
            )
        n_leader = sum(g["dot"] is not None for g in self.enc_labels)
        print(
            f"room labels: {len(numbers)} {ROOM_NUMBER_RE.pattern!r} texts, {n_no_box} without a box, "
            f"{len(self.enc_labels)} grouped ({n_leader} with a leader), "
            f"{len(self.unclassified_room_labels)} unclassified; "
            f"{len(dot_path) - len(used_dots)} of {len(dot_path)} end dots not in a label"
        )

    def _index_oval_heads(self) -> None:
        """Arrowhead candidates of oval tags: filled paths that are one closed triangle of 3 l items.

        The longest side is at most OVAL_HEAD_MAX_LEN and at least OVAL_HEAD_MIN_ASPECT times the
        shortest; the tip is the vertex opposite the shortest side. Sets oval_head_path (index into
        self.paths), oval_head_tip (n, 2) and oval_head_angle (base midpoint to tip, degrees, page space).
        """
        head_path, tips, angles = [], [], []
        for i, path in enumerate(self.paths):
            items = path["items"]
            if "f" not in path["type"] or len(items) != 3 or any(item[0] != "l" for item in items):
                continue
            starts = np.array([tuple(item[1]) for item in items])
            ends = np.array([tuple(item[2]) for item in items])
            if np.linalg.norm(ends - np.roll(starts, -1, axis=0), axis=1).max() > SUBPATH_TOL:
                continue  # not one closed triangle
            sides = np.linalg.norm(ends - starts, axis=1)  # side n runs from vertex n to vertex n + 1
            if sides.max() > OVAL_HEAD_MAX_LEN or sides.max() < OVAL_HEAD_MIN_ASPECT * sides.min():
                continue
            n = int(sides.argmin())
            base, tip = (starts[n] + ends[n]) / 2, starts[(n + 2) % 3]
            head_path.append(i)
            tips.append(tip)
            angles.append(math.degrees(math.atan2(tip[1] - base[1], tip[0] - base[0])))
        self.oval_head_path = np.array(head_path, dtype=int)
        self.oval_head_tip = np.array(tips, dtype=float).reshape(-1, 2)
        self.oval_head_angle = np.array(angles, dtype=float)

    def _oval_segments(self, bbox: Polygon) -> np.ndarray:
        """Segments of the closed chains around a text bbox: candidates for the oval of an oval tag.

        Candidates are straight segments at most OVAL_SEG_MAX_LEN long, inside bbox inflated by
        OVAL_MAX_GAP and not touching it. Segments with a free end (no other candidate's end within
        JOIN_TOL) are dropped until none is left, which removes leader pieces and stray wall
        pieces but keeps every closed chain, including chains drawn on top of each other.
        """
        search = shapely.box(*(np.array(bbox.bounds) + [-OVAL_MAX_GAP, -OVAL_MAX_GAP, OVAL_MAX_GAP, OVAL_MAX_GAP]))
        cand = self.seg_tree.query(search, predicate="contains")
        cand = cand[(self.seg_kind[cand] == "l") & (self.seg_len[cand] <= OVAL_SEG_MAX_LEN)]
        cand = cand[~shapely.intersects(self.seg_geoms[cand], bbox)]
        while len(cand):
            ends = shapely.points(self.seg_xy[cand].reshape(-1, 2))  # ends 2n and 2n + 1 belong to cand[n]
            ei, ej = STRtree(ends).query(ends, predicate="dwithin", distance=JOIN_TOL)
            other = ei // 2 != ej // 2
            joined = np.zeros(len(ends), dtype=bool)
            joined[ei[other]] = True
            free = ~joined.reshape(-1, 2).all(axis=1)
            if not free.any():
                break
            cand = cand[~free]
        return cand

    def get_oval_tags(self) -> None:
        """Group each oval tag: number, its mask, oval segments, leader and arrowhead.

        The oval is the set of closed chains around a text line matching OVAL_NUMBER_RE
        (_oval_segments) whose convex hull contains the text bbox, with every segment within
        OVAL_RING_TOL of the hull outline. The oval is drawn twice with overlapping segments, so
        no linemerge or polygonize. Text without such chains is not an oval tag. The number's mask
        is a fill whose bounds match the hull's within OVAL_MASK_TOL. The leader (_label_leader,
        OVAL_LEADER_MAX_TURN) runs from the oval to the tip of an arrowhead (_index_oval_heads), and
        its last piece is parallel to the arrowhead (ANGLE_TOL). An arrowhead reached from more than
        one oval leaves all of them unclassified. Membership is decided by geometry only.

        Appends to self.enc_labels (so call get_room_labels() first) groups {kind ("oval_tag"),
        text, rect, leader_rect, items}: rect is the minimum rotated rectangle of the oval (Polygon),
        leader_rect the bounding [x0, y0, x1, y1] of the leader, and items sorted indices into
        self.paths (page space, pt). Sets self.unclassified_oval_tags, a list of {text, bbox (the
        oval's), reason}. Uses only self.paths and the text, so it does not need parse().
        """
        if not hasattr(self, "unclassified_room_labels"):
            raise RuntimeError("call get_room_labels() before get_oval_tags()")
        if any(g["kind"] == "oval_tag" for g in self.enc_labels):
            raise RuntimeError("get_oval_tags() already ran")
        self._index_segments()
        self._index_masks()
        self._index_oval_heads()
        head_segs = np.array([s for p in self.oval_head_path for s in self.path_segs[int(p)]], dtype=int)

        self.unclassified_oval_tags = []
        numbers = np.flatnonzero([OVAL_NUMBER_RE.fullmatch(t) is not None for t in self.word_texts])
        cands = []  # tags whose parts were all found, before arrowhead ownership and path checks
        n_no_oval = 0
        for k in numbers:
            text, bbox = str(self.word_texts[k]), self.word_bboxes[k]
            oval = self._oval_segments(bbox)
            hull = shapely.convex_hull(shapely.multilinestrings(self.seg_geoms[oval])) if len(oval) else None
            if hull is None or hull.geom_type != "Polygon" or not hull.contains(bbox):
                n_no_oval += 1
                continue
            unclassified = {"text": text, "bbox": [round(float(v), 3) for v in hull.bounds]}
            reason = None
            off_ring = shapely.distance(self.seg_geoms[oval], hull.exterior).max()
            if off_ring > OVAL_RING_TOL:
                reason = f"segments around the number are not one oval ({off_ring:.2f} pt off its hull)"
            elif self.word_angles[k] != 0:
                reason = "number not horizontal"

            masks: list[int] = []
            if reason is None:
                cand = self.mask_tree.query(hull, predicate="intersects")
                off = np.abs(shapely.bounds(self.mask_polys[cand]) - np.array(hull.bounds)).reshape(-1, 4)
                masks = self.mask_path[cand[off.max(axis=1, initial=0.0) <= OVAL_MASK_TOL]].tolist()
                if len(masks) != 1:
                    reason = f"{len(masks)} candidate masks {masks} for {text!r}"

            leader = None
            if reason is None:
                exclude = np.concatenate([oval, head_segs])
                tips = self.oval_head_tip
                leader = self._label_leader(hull, exclude, tips, np.zeros(len(tips)), OVAL_LEADER_MAX_TURN)
                if leader is None:
                    reason = "no leader to an arrowhead"
                elif isinstance(leader, str):
                    reason = leader
                elif angle_diff(self.seg_angle[leader[0][-1]], self.oval_head_angle[leader[1]]) > ANGLE_TOL:
                    reason = "last leader piece not parallel to its arrowhead"
            if reason is not None:
                self.unclassified_oval_tags.append({**unclassified, "reason": reason})
                continue
            cands.append((text, unclassified, hull, oval, masks, leader))

        reached = defaultdict(int)  # arrowhead -> ovals whose leader reaches it
        for *_, (_, head) in cands:
            reached[head] += 1
        claimed: dict[int, str] = {}  # path index -> owner, for room labels and oval tags
        for g in self.enc_labels:
            claimed.update({p: f"room label {g['text']}" for p in g["items"]})
        used_heads: set[int] = set()
        n_groups = 0
        for text, unclassified, hull, oval, masks, (chain, head) in cands:
            if reached[head] > 1:
                reason = f"arrowhead reached from {reached[head]} ovals"
                self.unclassified_oval_tags.append({**unclassified, "reason": reason})
                continue
            paths, partial = self._group_paths(np.concatenate([oval, chain]).astype(int))
            if partial:
                reason = f"paths {partial} have items outside the group"
                self.unclassified_oval_tags.append({**unclassified, "reason": reason})
                continue
            items = sorted(set(paths + masks + [int(self.oval_head_path[head])]))
            taken = sorted({claimed[p] for p in items if p in claimed})
            if taken:
                reason = f"paths already used by {taken}"
                self.unclassified_oval_tags.append({**unclassified, "reason": reason})
                continue
            claimed.update({p: f"oval tag at {unclassified['bbox']}" for p in items})
            used_heads.add(head)
            leader_paths = sorted({int(self.seg_path[s]) for s in chain})
            self.enc_labels.append(
                {
                    "kind": "oval_tag",
                    "text": text,
                    "rect": shapely.minimum_rotated_rectangle(hull),
                    "leader_rect": union_rect([self.paths[p]["rect"] for p in leader_paths]),
                    "items": items,
                }
            )
            n_groups += 1
        print(
            f"oval tags: {len(numbers)} {OVAL_NUMBER_RE.pattern!r} texts, {n_no_oval} without an oval, "
            f"{n_groups} grouped, {len(self.unclassified_oval_tags)} unclassified; "
            f"{len(self.oval_head_path) - len(used_heads)} of {len(self.oval_head_path)} arrowheads not in an oval tag"
        )

    def draw(self, out_dir: Path) -> None:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, "enc_labels"):
            raise RuntimeError("call get_room_labels() before draw()")
        if not hasattr(self, "unclassified_oval_tags"):
            raise RuntimeError("call get_oval_tags() before draw()")
        label_geoms = [g["rect"] for g in self.enc_labels]
        label_geoms += [shapely.box(*g["leader_rect"]) for g in self.enc_labels if g["leader_rect"] is not None]
        label_geoms += [g["dot"] for g in self.enc_labels if g["kind"] == "room_label" and g["dot"] is not None]
        display_geoms(self.page, label_geoms, debug_dir / "labels.pdf")
        display_geoms(self.page, self.grid_lines, debug_dir / "grid_lines.pdf")

        # Copy the page so self.page stays untouched; insert_pdf keeps mediabox and rotation.
        out = pymupdf.open()
        out.insert_pdf(self.page.parent, from_page=self.page.number, to_page=self.page.number)
        if not hasattr(self, "measurements"):
            raise RuntimeError("call get_measurements() before draw()")
        if not hasattr(self, "tags"):
            raise RuntimeError("call get_tags() before draw()")
        if not hasattr(self, "align_markers"):
            raise RuntimeError("call get_align_markers() before draw()")
        if not hasattr(self, "callouts"):
            raise RuntimeError("call get_callouts() before draw()")
        grid = [style_key(path) == self.grid_key for path in self.paths]
        meas = {i for group in self.measurements for i in group["items"]}
        tags = {i for group in self.tags for i in group["items"]}
        align = {i for group in self.align_markers for i in group["items"]}
        callouts = {i for group in self.callouts for i in group["items"]}
        rooms = {i for group in self.enc_labels if group["kind"] == "room_label" for i in group["items"]}
        ovals = {i for group in self.enc_labels if group["kind"] == "oval_tag" for i in group["items"]}
        n_rooms = sum(group["kind"] == "room_label" for group in self.enc_labels)
        drop = [
            g or i in meas or i in tags or i in align or i in callouts or i in rooms or i in ovals
            for i, g in enumerate(grid)
        ]
        remove_items(out[0], self.paths, drop, drop_text=True)
        out[0].set_cropbox(pymupdf.Rect(self.drawing_area.bounds))
        out.save(out_dir / "drawing.pdf", garbage=3, deflate=True)
        print(f"{out_dir}: removed all text, {sum(grid)} grid line paths, {len(self.measurements)} "
              f"measurement groups ({len(meas)} paths), {len(self.tags)} tags ({len(tags)} paths), "
              f"{len(self.align_markers)} ALIGN markers ({len(align)} paths), "
              f"{len(self.callouts)} callouts ({len(callouts)} paths), "
              f"{n_rooms} room labels ({len(rooms)} paths), "
              f"and {len(self.enc_labels) - n_rooms} oval tags ({len(ovals)} paths), "
              f"cropbox {tuple(round(v, 2) for v in out[0].cropbox)}")
        self.write_unclassified(out_dir / "unclassified.yaml")
        # TODO@Claude Remove the remaining annotations as whole groups: title block, legend, enclosing
        # labels. (All text, grid lines, measurements, tags, ALIGN markers, callouts, room labels and
        # oval tags are done.)
        # Delete specific drawing items; never redact a bbox. Unclassifiable items stay (unclassified.yaml).
        # Save the remaining primitive geoms in self.prim (label geometry stays in self.enc_labels).


    def write_unclassified(self, path: Path) -> None:
        """Write the measurement texts, tags, ALIGN markers, callouts, room labels and oval tags that
        could not be grouped (left in the drawing) as YAML."""
        items = [
            {"kind": "measurement_text", "text": u["text"], "bbox": u["bbox"], "reason": u["reason"]}
            for u in self.unclassified_measurements
        ] + [
            {"kind": "tag", "text": u["text"], "bbox": u["bbox"], "reason": u["reason"]}
            for u in self.unclassified_tags
        ] + [
            {"kind": "align_marker", "text": u["text"], "bbox": u["bbox"], "reason": u["reason"]}
            for u in self.unclassified_align_markers
        ] + [
            {"kind": "callout", "text": u["text"], "bbox": u["bbox"], "reason": u["reason"]}
            for u in self.unclassified_callouts
        ] + [
            {"kind": "room_label", "text": u["text"], "bbox": u["bbox"], "reason": u["reason"]}
            for u in self.unclassified_room_labels
        ] + [
            {"kind": "oval_tag", "text": u["text"], "bbox": u["bbox"], "reason": u["reason"]}
            for u in self.unclassified_oval_tags
        ]
        doc = {
            "source": self.page.parent.name,
            "page": self.page.number,
            "coordinate_space": "PyMuPDF page space, origin top-left, y down",
            "units": "pt",
            "items": items,
        }
        path.write_text(yaml.safe_dump(doc, sort_keys=False))

    def encode_prims(self, annot_path):
        '''
        TODO@Claude
        Input: self.prim [Shapely geometries of the architectural drawing only (no annotations)]
        Input: annot_path [annotations/<pdf stem>.xml, one per PDF; CVAT XML 1.1 with rotated boxes; not COCO]
        Output: OUT_DIR/<input-stem>/<input-stem>_s2.npz, one per page, in the parse_svg_v5.py format:
            per primitive: args (8 floats: the 4 sampled (x, y) points), lengths, commands (0-3),
            widths (> 0), semanticIds, instanceIds, layerIds, rgb (stroke color);
            per instance: obj_cts [cx, cy, 0, instanceId], boxes [x1, y1, x2, y2, semanticId],
            both from the extent of the instance's primitives;
            plus width, height. See Contracts in CLAUDE.md.

        Sample all primitives in self.prim as follows:
            * 4 points sampled at 0, 1/3, 2/3, 1 of the length for lines/arcs
            * circles sampled at angles 0, pi/2, pi, 3pi/2

        Labels: each <box label xtl ytl xbr ybr rotation> is one instance, in pixels, rotated about its
        center. Scale to PDF points by page.rect.width / width and page.rect.height / height.
        semanticIds = 0-based index of label in SVG_CATEGORIES (35 = background); instanceIds < 0 = none.
        Match each <image> to its PDF by the stem in name; log unmatched ones.
        Open decisions (ask first): which box a primitive belongs to, and how unannotated stuff
        classes (walls) are labeled.
        Format reference: parse_svg_v5.py (not parse_svg.py). Don't write a .npz loader.
        '''
        





