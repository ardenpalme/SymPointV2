import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import polygonize


def bezier_points(control, count=20):
    t = np.linspace(0.0, 1.0, count)[:, None]
    p0, p1, p2, p3 = control
    return ((1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3)

def is_perpendicular(corner, p, q, tolerance):
    ux, uy = p[0] - corner[0], p[1] - corner[1]
    vx, vy = q[0] - corner[0], q[1] - corner[1]
    return abs(ux * vx + uy * vy) <= tolerance * math.hypot(ux, uy) * math.hypot(vx, vy)


def find_rectangles(lines, decimals=2, tolerance=0.02):
    neighbors = defaultdict(set)
    line_edges = []
    for line in lines:
        a, b = [(round(p[0], decimals), round(p[1], decimals)) for p in (line.coords[0], line.coords[-1])]
        line_edges.append(frozenset((a, b)))
        if a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)

    rectangles = []
    used_edges = set()
    for a, near in neighbors.items():
        for b, d in combinations([p for p in near if p > a], 2):
            if not is_perpendicular(a, b, d, tolerance):
                continue
            for c in neighbors[b] & neighbors[d]:
                if c > a and is_perpendicular(b, a, c, tolerance) and is_perpendicular(d, a, c, tolerance):
                    rectangles.append(Polygon([a, b, c, d]))
                    used_edges.update(frozenset(edge) for edge in [(a, b), (b, c), (c, d), (d, a)])

    return rectangles, [line for line, edge in zip(lines, line_edges) if edge not in used_edges]

def find_enclosed_circles(lines, min_segments=8, min_convexity=0.99, min_rectangularity=0.9, margin=0.5):
    noded = shapely.unary_union(lines)
    faces = list(polygonize(noded)) 

    circles = []
    bands = []
    for face in faces:
        polygon = Polygon(face.exterior) # ignore holes

        convexity = polygon.area / polygon.convex_hull.area
        if convexity < min_convexity: continue # Skip concave regions.

        # Ratio is about 1.0 for a rectangle and about 0.785 for an ellipse.
        rectangularity = polygon.area / polygon.minimum_rotated_rectangle.area
        # Skip rectangle-like regions such as rooms.
        if rectangularity >= min_rectangularity: continue

        # An outline with N points has N - 1 sides.
        segment_count = len(polygon.exterior.coords) - 1
        # Skip shapes with too few sides to be round.
        if segment_count < min_segments: continue

        # Store the filled shape, so text inside it counts as intersecting.
        circles.append(polygon)
        # Store a band of width `margin` on both sides of the outline.
        bands.append(polygon.exterior.buffer(margin))

    # Merge all bands into one area.
    used_area = shapely.unary_union(bands)
    # True for every input line that lies entirely inside a band.
    is_used = shapely.within(lines, used_area)
    # Keep the lines that did not draw a detected circle.
    kept = [line for line, used in zip(lines, is_used) if not used]

    # Detected circles and the remaining segments.
    return circles, kept
