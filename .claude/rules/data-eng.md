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
- **Tag**: diamond with a code (e.g. `G03`) and a leader. The leader ends on a wall in a short line
  stroked with a large width and round caps (not a filled shape).
- **ALIGN marker**: bracket with two solid arrowheads pointing at wall faces, plus the text `ALIGN`.
- **Grid line**: long thin gray line across the drawing (the parser separates them by a gray style key).
- Measurements, callouts, tags, ALIGN markers, grid lines, and all text are annotations: removed as
  whole groups. Examples: `.claude/agents/assets/`.
- **Primitive**: one architectural line, arc, or circle left after all annotations are removed.
- **Style key**: `(color, width)` of a path; acts like a CAD layer. Differs per file.

## Outputs: `OUT_DIR/<input-stem>/` (`OUT_DIR` is set in `main.py`)
- `title_block.pdf`, `legend.pdf` + `legend.yaml`: cropped from the source page.
- `drawing.pdf`: drawing area with all annotations removed; nothing else changed. Keep the source
  page's mediabox and rotation and crop with a cropbox, so it can be diffed against the source.
- `measurements.yaml` (endpoints, text as written, length in pt), `rooms.yaml` (text, position, bbox).
- `<input-stem>_s2.npz`: primitives from `self.prim`, in the `parse_svg_v5.py` format (see Contracts in
  CLAUDE.md). `semanticIds` and `instanceIds` come from the boxes in `annotations/<pdf stem>.xml`.
- Label geometry (shapes enclosing room labels) stays in `self.enc_labels`; it never goes into `self.prim`.
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
- Closed annotation shapes mix item kinds: a callout "circle" is Bézier curves plus line segments, and
  its black part is a filled path. Detect closed shapes after merging curves and lines together.
- ALIGN arrowheads are bounded by axis-aligned `re` items in paths with `width` or `color` None, which
  `parse()` currently skips. Keep those paths, and convert each arrowhead to a Shapely polygon with
  `shapely.minimum_rotated_rectangle`.
- The tag leader's end bar is a line item: find it by its large stroke width and round caps
  (`lineCap` in `get_drawings()`), not by shape.

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
