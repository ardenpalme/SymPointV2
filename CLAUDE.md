# Panoptic symbol spotting on architectural floor plans (SymPoint-V2 based)

Goal: run the `svgnet/` model, trained on FloorPlanCAD, on real A0 floor plan PDFs.

## Pipeline
1. **FloorPlanCAD (training data)**: `dataset/svg` is a symlink to the cluster copy; it may not resolve
   locally. `*.cocojson` are FloorPlanCAD bbox ground truth for evaluation
   (`gen_coco_ann.py`, `gen_coco_det.py`, `coco_eval.py`).
2. **Model**: `svgnet/` (data, model, evaluation), config `configs/svg/svg_pointT.yaml`, entry points in `tools/`.
3. **A0 PDFs (current focus)**: `main.py` + `data_eng/parser.py` clean a whole A0 page (remove title
   block, legend, grid lines, and all annotations) and write the remaining primitives as `*_s2.npz`.
   Labels come from `annotations/<pdf stem>.xml` (one CVAT XML per PDF, rotated boxes).
   Rules: `.claude/rules/data-eng.md`.
4. **Sliding-window inference (not built)**: A0 pages are far larger than FloorPlanCAD tiles, so they are
   split into overlapping windows and predictions are aggregated. A previous attempt (core cells plus
   margins, each instance owned by the cell containing its bbox center) did not work; cause unknown.

## Contracts
- `SVGDataset.load` in `svgnet/data/svg3.py` reads `*_s2.json` with keys `width`, `height`, `args`
  (8 floats per primitive: 4 (x, y) points), `lengths`, `commands` (0–3), `widths` (must be > 0),
  `semanticIds`, `instanceIds`, `layerIds`. Coordinates are normalized by `width` and `height`.
- Stage 3 writes one `<input-stem>_s2.npz` per page with those keys and per-primitive layout, matching
  `parse_svg_v5.py` (the canonical reference). No loader reads `.npz` yet. <!-- TODO --> Don't write
  or change a loader; if a task needs one, stop and ask.
- The model was trained on FloorPlanCAD tiles: windows need comparable extent and primitive density.
- Semantic ids are 0-based indices into `SVG_CATEGORIES` (35 = background), not its 1-based `id` field.
  Instance id < 0 means no instance. The class list is duplicated across files; never change it.
- If the PDF pipeline and the model disagree, stop and ask. Never patch the model or loaders to fit.

## Environment
- Conda env `sp` (Python 3.9 <!-- TODO: confirm -->), activated before Claude Code starts.
  If `sys.prefix` doesn't end in `envs/sp`, prefix commands with `conda run -n sp --no-capture-output`.
- No dependency list in the repo. Ask before installing anything.
- Python 3.9: no `match`, no runtime `X | Y` unions (use `from __future__ import annotations`).

## Commands
- Clean data: `python main.py`. Input and output folders are `DATA_DIR` and `OUT_DIR` in its `main()`.
- Lint: `ruff check <files you changed>`. Run `ruff format` only on files you created.
- No test suite or type checker yet. Don't invent commands for them. <!-- TODO -->
- Training and testing: `tools/train_dist.sh`, `tools/test_dist.sh`, on the Slurm cluster. <!-- TODO: args -->

## Boundaries
- Never launch training, inference, or Slurm jobs, or rebuild the CUDA ops. Give me the command instead.
- `dataset/` and `modules/pointops/` (prebuilt CUDA extension) are read-only.
- The format reference is `parse_svg_v5.py`. Don't use `data_eng/parse_svg.py` or `parse_svg_v2.py`
  for stage 3 output.
- Never edit or import `decoder copy*.py` or `*.bak`.
- `svgnet/data/` has several loader variants (`svg.py`, `svg2.py`, `svg3.py`, `no_svg*.py`). Before editing
  or importing one, trace which is live from the config and imports.
- Model code (`svgnet/`, `modules/`, `tools/`) is upstream research code: change it only when the task requires it.

## Known issues (separate tasks; don't fix as a side effect)
- `main.py` processes only `os.listdir(DATA_DIR)[0]` and page 0, and imports via `sys.path` hacks.
- No `pyproject.toml`; packages aren't installable.

## Tasks
- Small tasks are `TODO@Claude` comments. One per request; remove only when implemented and verified.
- Ambiguous or conflicting TODO: stop and suggest `/grill-with-docs`.
- Multi-file work: `/to-spec`, `/to-tickets`, `/implement`. Tickets live in <location>. <!-- TODO -->
