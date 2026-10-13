from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pymupdf
import shapely
from shapely import LineString, Polygon, STRtree
from shapely.ops import linemerge, polygonize
from tqdm import tqdm

from .geom import bezier_points, find_enclosed_circles, find_rectangles
from .utils import display_geoms, remove_items

THRSH = 3.0
ANGLE_TOL = 3.0


def style_key(path: dict) -> tuple | None:
    """(color, width) of a get_drawings() path, rounded to 4 decimals; None if either is None."""
    path_width, path_color = path.get("width"), path.get("color")
    if path_width is None or path_color is None:
        return None
    return (tuple(round(ele, 4) for ele in path_color), round(path_width, 4))


class FloorPlanParser:
    def __init__(self, page) -> None:
        self.page = page
        self.paths = page.get_drawings()
        self.blocks = page.get_text("dict")["blocks"]

    def parse(self):
        lines, curves, rects = defaultdict(list), defaultdict(list), defaultdict(list)
        for path in tqdm(self.paths):
            key = style_key(path)
            if key is None:
                continue

            for item in path["items"]:
                kind = item[0]
                if kind in ["re", "qu"]:
                    q = item[1].quad if item[0] == "re" else item[1]
                    rect = Polygon([q.ul, q.ur, q.lr, q.ll, q.ul])
                    rects[key].append(rect)
                    
                elif kind == "l":
                    line = LineString([item[1], item[-1]])
                    lines[key].append(line)
                    
                elif kind == "c":
                    nodes = np.array(item[1:])
                    if np.ptp(nodes, axis=0).max() == 0: continue
                    curve = shapely.LineString(bezier_points(nodes))
                    curves[key].append(curve)
                    
                else:
                    raise ValueError("unhandled drawing", item)

        geoms_dict = copy.deepcopy(lines)
        geoms_dict.update(curves)

        # TODO@CLaude: These are only valid regions for one of the files, 
        # To build a robust pipeline, you will need to pre-process files where the drawing area and the 
        # grid lines do not follow this exactly
        self.drawing_area = polygonize(lines[((0.0, 0.0, 0.0), 1.68)])[0] 
        self.grid_key = ((0.498, 0.498, 0.498), 0.24)
        self.grid_lines = lines[self.grid_key]
        lines.pop(((0.0, 0.0, 0.0), 1.68))
        lines.pop(self.grid_key)

        rl = []
        for val in lines.values():
            rl.extend(val)
        merged_lines = linemerge(rl)
        raw_lines = np.array(list(merged_lines.geoms))

        cs = []
        for val in curves.values():
            cs.extend(val)
        merged_curves = linemerge(cs)
        curves_list = list(merged_curves.geoms)

        self.circles = []
        for curve in curves_list:
            if(curve.is_ring):
                self.circles.append(Polygon(curve.coords))

        self.rects, rem_ls = find_rectangles(raw_lines)
        self.enc_cir, ls = find_enclosed_circles(rem_ls)

        self.conv_sh = np.array(self.enc_cir + self.circles + self.rects)
        self.conv_tree = STRtree(self.conv_sh)

        self.lines = ls
        self.lines_tree = STRtree(self.lines)

        word_bboxes = []
        word_texts = []
        angles = []

        for block in self.blocks:
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

        self.word_bboxes = np.array(word_bboxes)
        self.word_texts = np.array(word_texts)
        self.word_angles = np.array(angles)

        word_idx, sh_idx = self.conv_tree.query(self.word_bboxes, predicate="intersects")
        overlap = shapely.area(shapely.intersection(self.conv_sh[sh_idx], self.word_bboxes[word_idx]))
        word_share = overlap / shapely.area(self.word_bboxes[word_idx])
        shape_share = overlap / shapely.area(self.conv_sh[sh_idx])
        keep = (word_share > 0.40) & (shape_share > 0.1)
        self.enc_labels = self.conv_sh[sh_idx[keep]]

    def get_measurements(self):
        mask_align = np.strings.find(self.word_texts, 'ALIGN') == 0
        mask_meas = np.strings.find(self.word_texts, 'mm') >= 0
        mask = (mask_meas | mask_align)
        w_bbox_dist = self.word_bboxes[mask]  # noqa: F841

        bbox_idx, line_idx = self.lines_tree.query(self.word_bboxes[mask], predicate="dwithin", distance=THRSH)
        ends = np.array([[line.coords[0], line.coords[-1]] for line in self.lines])
        dx = ends[:, 1, 0] - ends[:, 0, 0]
        dy = ends[:, 1, 1] - ends[:, 0, 1]
        line_angles = np.degrees(np.arctan2(dy, dx))
        diff = np.abs(line_angles[line_idx] - self.word_angles[bbox_idx]) % 180
        diff = np.minimum(diff, 180 - diff)
        parallel = diff <= ANGLE_TOL
        line_idx, word_idx = line_idx[parallel], bbox_idx[parallel]  # noqa: F841

        # TODO@Claude finish this function to group the architectural measurements lines next to or pointed to from w_bbox_dist
        # sometimes the measurement lines are parallel to w_bbox_dist, and sometimes there is a pointer from the w_bbox_dist boxes to the measurement lines

    def draw(self, out_dir: Path) -> None:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        display_geoms(self.page, self.enc_labels, debug_dir / "labels.pdf")
        display_geoms(self.page, self.grid_lines, debug_dir / "grid_lines.pdf")

        # Copy the page so self.page stays untouched; insert_pdf keeps mediabox and rotation.
        out = pymupdf.open()
        out.insert_pdf(self.page.parent, from_page=self.page.number, to_page=self.page.number)
        drop = [style_key(path) == self.grid_key for path in self.paths]
        remove_items(out[0], self.paths, drop, drop_text=True)
        out[0].set_cropbox(pymupdf.Rect(self.drawing_area.bounds))
        out.save(out_dir / "drawing.pdf", garbage=3, deflate=True)
        print(f"{out_dir}: removed all text and {sum(drop)} grid line paths, "
              f"cropbox {tuple(round(v, 2) for v in out[0].cropbox)}")
        # TODO@Claude Remove the remaining annotations as whole groups: title block, legend, enclosing
        # labels, measurement groups, callouts, tags, ALIGN markers. (All text and grid lines are done.)
        # Delete specific drawing items; never redact a bbox. Unclassifiable items stay (unclassified.yaml).
        # Save the remaining primitive geoms in self.prim (label geometry stays in self.enc_labels).


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
        





