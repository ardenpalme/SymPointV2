---
paths:
  - "data_eng/**/*.py"
  - "**/main.py"
---
# Stage 3: A0 floor plan cleaning

Splits vector floor plan PDFs into title block, legend, room labels, measurements, and a clean
drawing area, then encodes the remaining primitives for the model.

## Terms
- **Measurement**: dimension line + ticks at 45° to the line (or arrows) at both ends + perpendicular
  extension lines + text such as `1691 mm` + sometimes a leader.
- **Leader**: straight, bent, or curved line from displaced text to its dimension line, sometimes with
  an arrowhead. Not every arrow is a leader; door and window direction arrows are architecture.
- **Reference callout**: circle with a code (e.g. `A-4.03.51`) and a solid black lower part.
- **Tag**: diamond with a code (e.g. `G03`) and a leader. The leader ends on a wall in a short,
  very wide stroke (the "end bar").
- **ALIGN marker**: bracket with two solid arrowheads pointing at wall faces, plus the text `ALIGN`.
- **Room label**: room name in bold capitals (one or more lines, e.g. `PRINT / COPY`) above a
  rectangle containing the room number (e.g. `03-A36`). Sometimes a leader, straight or bent,
  runs from the rectangle to a solid black dot inside the room. Leaders often cross walls.
  Examples: `.claude/agents/assets/room_label_*.png`.
- **Oval tag**: a number (e.g. `110`) inside an oval outline, with a leader (often bent once)
  ending in a solid black arrowhead that touches a wall or window.
  Examples: `.claude/agents/assets/oval_tag_*.png`.
- **Grid line**: long thin gray line across the drawing (the parser separates them by a gray style key).
- Measurements, callouts, tags, ALIGN markers, room labels, oval tags, grid lines, and all text are
  annotations: removed as whole groups. Examples: `.claude/agents/assets/`.
- **Primitive**: one architectural line, arc, or circle left after all annotations are removed.
- **Style key**: `(color, width)` of a path; acts like a CAD layer. Differs per file.

## Outputs: `OUT_DIR/<input-stem>/` (`OUT_DIR` is set in `main.py`)
- `title_block.pdf`, `legend.pdf` + `legend.yaml`: cropped from the source page.
- `drawing.pdf`: drawing area with all annotations removed; nothing else changed. Keep the source
  page's mediabox and rotation and crop with a cropbox, so it can be diffed against the source.
- `measurements.yaml` (endpoints, text as written, length in pt), `rooms.yaml` (text, position, bbox).
- `<input-stem>_s2.npz`: primitives from `self.prim`, in the `parse_svg_v5.py` format (see Contracts in
  CLAUDE.md). `semanticIds` and `instanceIds` come from the boxes in `annotations/<pdf stem>.xml`.
- Room label groups (`kind: room_label`, `text`, `name`, `rect`, `leader_rect`, `dot`, `items`) and oval
  tag groups (`kind: oval_tag`, `text`, `rect`, `leader_rect`, `items`) are stored in `self.enc_labels`;
  they never go into `self.prim`. Filter by `kind`.
- YAML starts with `source`, `page`, `coordinate_space`, `units`. Debug renders go in `debug/`.
  No new files or fields without asking.

## Annotations: `annotations/<pdf stem>.xml`
- `annotations/` holds hand-made labels, one file per PDF, named like the PDF
  (`tmp_dataset/X.pdf` → `annotations/X.xml`). It is read-only.
- One `<image>` per page, in page order; if the counts differ, raise. Each `<image>` is a
  raster of the page, with `width` and `height` in pixels. Its `name` is the raster's file
  name and need not match the PDF; don't use it for matching.
- Format: CVAT XML version 1.1 (root `<annotations>`). Not COCO JSON; don't use pycocotools.
- Each object is `<box label xtl ytl xbr ybr rotation>`: the unrotated box in pixels, rotated by
  `rotation` degrees about its center (missing attribute = 0). Confirm the rotation direction by
  rendering a few boxes over the page before relying on it.
- Pixels to PDF points: scale by `page.rect.width / width` and `page.rect.height / height`
  (the first file is 9362 × 6623 px for an A0 sheet, about 200 DPI). Never hardcode the DPI.
- `label` matches a `SVG_CATEGORIES` name exactly, case included (`Wardrobe`, `TV cabinet`);
  semantic id = its 0-based index. Each box is one instance.
- Only "thing" classes are annotated so far (doors, windows, stairs, elevators). Walls, curtain walls,
  railings, and other "stuff" classes have no boxes.

## Open design decisions (ask; don't decide)
- When a primitive belongs to a box (center inside, fully inside, or overlap fraction), and which box
  wins when boxes overlap.
- Primitives of unannotated stuff classes (walls): label them background (35), or exclude them.
- Window size and overlap in page units, and how to match FloorPlanCAD tile scale.
- Whether windowing runs on cleaned primitives or on the PDF page.
- How instances crossing window boundaries are merged.
- A PDF without an annotation file: skip it, or write it with all primitives as background (35) and
  instance -1 (e.g. for inference-only drawings).
- Filled shapes such as wall poché: encode their clip outlines as primitives, or encode strokes only.

## Data rules
- `DATA_DIR` PDFs are read-only. Process every PDF in sorted order, every page. Deterministic output.
- Remove annotations by deleting specific drawing items, never by redacting a bbox (erases walls).
- Unclassifiable items stay in the drawing and go in `unclassified.yaml`. Never guess.
- Drawings come from different offices. A heuristic checked on one file is not done; say which files you checked.

## Geometry
- PDF points, PyMuPDF page space (origin top-left, y down). Apply `page.rotation` consistently.
- Tolerances are module constants with units in a comment (`THRSH` 3.0 pt, `ANGLE_TOL` 3.0°).
- Never hardcode a style key; derive it per drawing and log which key was chosen and why.
- Vectorized Shapely 2 / STRtree queries; no Python loops over pairwise geometry.
- Unknown drawing item kinds raise; never skip silently.
- ALIGN arrowheads are clipped fills (see PDF encoding). Represent each arrowhead by its clip
  polygon, closed, as a Shapely polygon. Don't use the fill's `re`: it is only the bounding box and
  about twice the arrowhead's area.

## PDF encoding (observed in A-1-03-10 only; other offices may differ)
Facts:
- Most paths are a single segment: polylines, leaders, and many arcs are split into one-segment
  paths. Join pieces by shared endpoints (tolerance as a module constant) before detecting shapes.
  Exceptions: 62 stroked paths made only of curves (e.g. the callout circle, one path of 4 curves),
  6 paths mixing curves and lines, and 40 filled l polygons. 21 paths are both filled and stroked.
- Style keys don't separate annotation from architecture: walls, dimension and extension lines,
  leaders, and tag diamonds share black 0.36 or 0.24. Classify by geometry and relations (ticks,
  text, leaders), never by style key alone.
- `lineCap` is (1, 1, 1) on every stroke and None on fills. Don't use it.
- Most fill paths (1,647 of 1,687) are a single axis-aligned `re` item with color and width None.
  The other 40 fills are `l` polygons, including rotated text masks.
- Clipped fills: in `get_drawings(extended=True)`, a `clip` entry at level 0 followed directly by one
  level-1 entry means that entry is clipped. The clip's path is the visible shape (close it when
  building the polygon); the `re` is only its bounding box, and the clip's `scissor` is the tighter
  bbox. 1,158 of the 1,647 `re` fills are clipped this way, plus 5 strokes. No clip covers more than
  one entry.
- Example: each ALIGN arrowhead is a triangle clip (tip on the wall face) followed by a black `re`
  fill of the clip's bounding box.
- Text has a white mask fill underneath: an `re` for horizontal text, a 4-segment `l` polygon for
  rotated text. A mask belongs to its text: remove them together, or it keeps hiding architecture.
  Exception: room names have no mask; the room number's box `re` is its mask.
- Drawing order is only a weak hint: annotation strokes come late, fills (including ALIGN
  arrowheads) come early.
- The oval of an oval tag is drawn twice, as two closed chains of one-segment lines, not curves:
  21 segments, then 34 (55 per oval, all 9 tags). The chains are different polygonizations of the
  same ellipse, at most 0.14–0.24 pt apart, so they overlap visually but rarely exactly: within
  0.05 pt, 0–2 segment pairs per oval overlap; within 0.2 pt, 37–40 pairs (52–55 of 55 segments).
  All overlaps are between the two chains, never within a chain. The duplicate chain breaks linemerge
  and polygonize, so detect the oval as the cluster of segments around the number, or by its convex hull.

Signatures (widths are this file's values; derive them per drawing):
- Measurement: ticks are short 45° strokes of width 0.48 (about 4.3 × 4.3 pt on a horizontal line,
  6 pt axis-aligned on a 45° line). Dimension and extension lines are 0.36. A leader is a chain of
  short 0.36 segments from a text corner to the dimension line.
- Tag: the diamond is 4 diagonals of 12 × 12 pt over a white 24 × 24 pt `re`. The leader is one
  0.36 line. The end bar is a 5.6–8.2 pt stroke of width 1.92, the widest stroke on the page.
- ALIGN: the bracket is 3 lines of width 0.24. The arrowheads are black fill `re` items of
  2.3 × 8.9 pt. There's no text mask.
- Callout: the circle is one 0.36 path of 4 curves. Each black part is curve paths, a 7-segment line
  chain, and a black fill `re` (its bounding box).
- Grid lines: gray 0.498 at width 0.24, drawn as separate 6 pt and 24 pt pieces, not a dash pattern.
- Room label: one contiguous run in drawing order: dot, leader, 4 box edges, white `re`, the same
  4 edges again, then one text object with the name lines and the number (ArialNarrow,Bold 8.05 pt).
  All black 0.36, level 0, unclipped. Box 25.4–25.6 × 9.8–10.0 pt. Leader: 1–2 one-segment lines,
  23.5–75.4 pt, from the box edge (7) or the name bbox (3; one ends 2.2 pt below it) to the dot
  center. Dot: one `fs` path of 4 curves, black fill and stroke, 5.6–5.8 pt. 72 labels (69 distinct
  numbers), 10 with a leader, 10 dots on the page.
- Oval tag: one contiguous run in drawing order: arrowhead, leader, oval chain (21 segments),
  clip + white `re` (the number's mask), the oval again (34 segments). Black 0.36, level 0,
  unclipped, except the mask. Arrowhead: one `fs` path of 3 `l` items (closed triangle, black
  fill and stroke, first point is the tip), about 8.5 pt long with a 2.3 pt base. Not a clipped fill.
  Leader: 1–2 one-segment lines (7 of 9 bent), 16.5–37.7 pt in total, from the tip to the oval's
  right end (4) or 0.6–0.84 pt short of its top, bottom, or left (5). Mask: a clip of 21–25 `l` items
  tracing the oval, then a level-1 white `re` of 25.4–25.6 × 14.2 pt. Number: one span, ArialNarrow
  8.03 pt, black, horizontal. 9 oval tags on the page (108 ×4, 109 ×2, 110 ×2, 111). The other
  2 black 3-line `fs` paths (width 0.24, top right) are not oval tags.

Unverified (check before relying on them):
- That all 91 strokes of width 1.92 are tag end bars.
- That architecture never uses width 0.48 or 1.92.
- That the 489 unclipped `re` fills (mostly white, some green, black, light blue) are real rectangles.
- That the 1491 × 1491 pt black fill (clipped by a 163-segment polygon) is wall poché.
- That the 29 regular-weight, mostly 45°-rotated `NN-XNN` texts are door numbers (not room labels).
- That room label leaders start only on the box or the name bbox.

## Testing
- Deterministic helpers (sampling, geometry utils, writers): `tdd`.
- `*_s2.npz` output: test that its keys, dtypes, and array shapes match what `parse_svg_v5.py` writes
  and what `SVGDataset.load` in `svgnet/data/svg3.py` reads.
- Heuristics: golden files. Fixture PDFs in `tests/fixtures/`, expected outputs in `tests/expected/`.
  Report which expected outputs changed and why. Never edit `tests/expected/` to make tests pass.

## Verifying
- Run on all input PDFs. Then run the `render-checker` agent once per drawing (in parallel is fine),
  with output in `OUT_DIR/<input-stem>/debug/check/`. A change is done only when every drawing passes.
- If a check fails, fix and re-check, at most 3 rounds. Then stop and show me the failing tiles.
- Report counts per drawing: rooms, measurements, legend entries, primitives, unclassified items.
- After changing a classification heuristic, stop and show me `overview.png` for each drawing.
