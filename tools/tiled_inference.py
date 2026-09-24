"""
Tiled SymPointV2 inference for large PDF pages, merged back to page level.

Tiling
  The page is cut into a grid of CORE cells (stride s). Each tile = its core cell
  plus a margin m on every side (tile side T = s + 2m). A tile contains every
  primitive whose center lies in its window. Each tile is normalized by T, so
  symbol scale is the same in every tile and close to FloorPlanCAD.

Merge (exact partition, no double counting)
  - Every primitive center falls in exactly one core cell -> its semantic scores
    come from that cell's tile.
  - An instance predicted in a tile is kept only if the center of its bounding box
    lies in that tile's core cell. The same object seen by a neighbouring tile has
    its center in a different cell there, so it is dropped. Kept instances retain
    all their primitives, including those in the margin, so objects crossing a
    cell border stay whole (requires m > largest symbol size).

Usage
  PYTHONPATH=./ python tools/tiled_inference.py CFG CKPT --datadir dataset/pdfs --split test \
      --target 3000 --margin 0.25 --out tiled.npy \
      --cvat_xml annotations.xml --cvat_img_dir cvat_imgs --cvat_mode box
  Add --dry_run to only print the tiling for each page.
"""
import argparse, io, math, os, os.path as osp, time
from glob import glob

import numpy as np
import torch
import tqdm
import yaml
from munch import Munch
import pymupdf

from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.data.svg3 import PDFDataset
from svgnet.util import get_root_logger, load_checkpoint
from tools.inference import cvat_root, add_cvat_image, write_xml, to_np

NPZ_KEYS = ("coords", "lengths", "commands", "widths", "semanticIds", "instanceIds", "layerIds")
SKIP = {"wall", "curtail wall" , "row chairs", "railing"}

# ------------------------------------------------------------------ tiling
def prim_centers(coords):
    return coords.reshape(-1, 4, 2).mean(1)                      # (N, 2)


def choose_stride(centers, W, H, target, margin_frac):
    """Largest-to-smallest search: pick core stride s whose median primitives per
    tile window (core count * (1+2*margin_frac)^2) is closest to target."""
    target_core = target / (1 + 2 * margin_frac) ** 2
    best = None
    for s in np.geomspace(max(W, H), max(W, H) / 60, 120):
        nx, ny = max(1, math.ceil(W / s)), max(1, math.ceil(H / s))
        cx = np.clip((centers[:, 0] // s).astype(int), 0, nx - 1)
        cy = np.clip((centers[:, 1] // s).astype(int), 0, ny - 1)
        counts = np.bincount(cy * nx + cx, minlength=nx * ny)
        med = np.median(counts[counts > 0])
        err = abs(med - target_core)
        if best is None or err < best[0]:
            best = (err, float(s))
    return best[1]


def make_tiles(centers, W, H, s, m):
    """Yield (i, j, x0, y0, window_idx, owned_mask_within_window)."""
    nx, ny = max(1, math.ceil(W / s)), max(1, math.ceil(H / s))
    cell_x = np.clip((centers[:, 0] // s).astype(int), 0, nx - 1)
    cell_y = np.clip((centers[:, 1] // s).astype(int), 0, ny - 1)
    for j in range(ny):
        for i in range(nx):
            own = (cell_x == i) & (cell_y == j)
            if not own.any():
                continue
            # border cells extend to infinity so primitives outside [0,W]x[0,H] are covered
            xlo = -np.inf if i == 0 else i * s - m
            xhi = np.inf if i == nx - 1 else (i + 1) * s + m
            ylo = -np.inf if j == 0 else j * s - m
            yhi = np.inf if j == ny - 1 else (j + 1) * s + m
            win = ((centers[:, 0] >= xlo) & (centers[:, 0] < xhi) &
                   (centers[:, 1] >= ylo) & (centers[:, 1] < yhi))
            idx = np.nonzero(win)[0]
            yield i, j, i * s - m, j * s - m, idx, own[idx]


def cell_of(pt, s, nx, ny):
    return (int(np.clip(pt[0] // s, 0, nx - 1)), int(np.clip(pt[1] // s, 0, ny - 1)))


def tile_sample(z, idx, x0, y0, T, norm_ds):
    """Build one model input from a subset of primitives, reusing PDFDataset.load exactly."""
    coords = z["coords"].reshape(-1, 8)[idx].astype(np.float32).copy()
    coords[:, 0::2] -= x0
    coords[:, 1::2] -= y0
    buf = io.BytesIO()
    np.savez(buf, width=np.int64(round(T)), height=np.int64(round(T)), coords=coords,
             **{k: z[k][idx] for k in NPZ_KEYS if k != "coords"})
    buf.seek(0)
    return norm_ds.transform_test(*PDFDataset.load(buf, 0))


# ------------------------------------------------------------------ main
def get_args():
    p = argparse.ArgumentParser("tiled svgnet")
    p.add_argument("config"); p.add_argument("checkpoint")
    p.add_argument("--datadir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="tiled.npy")
    p.add_argument("--target", type=float, default=3000,
                   help="primitives per tile window; set to FloorPlanCAD median")
    p.add_argument("--stride", type=float, default=None,
                   help="core cell size in PDF units; overrides --target")
    p.add_argument("--margin", type=float, default=0.25, help="margin as fraction of stride")
    p.add_argument("--keep_thr", type=float, default=0.1, help="drop instances below this score")
    p.add_argument("--dry_run", action="store_true")
    # CVAT
    p.add_argument("--cvat_xml", default=None)
    p.add_argument("--cvat_img_dir", default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--score_thr", type=float, default=0.5)
    p.add_argument("--cvat_mode", choices=["box", "polyline"], default="box")
    return p.parse_args()


def main():
    args = get_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config)))
    logger = get_root_logger()
    files = sorted(glob(osp.join(args.datadir, args.split, "*_s2.npz")))
    logger.info(f"{len(files)} pages")

    norm_ds = PDFDataset.__new__(PDFDataset)           # only for transform_test
    norm_ds.data_norm = "mean"

    if not args.dry_run:
        model = svgnet(cfg.model).cuda(); model.eval()
        load_checkpoint(args.checkpoint, logger, model)
    num_classes = cfg.model.semantic_classes

    if args.cvat_xml:
        cvat = cvat_root(num_classes)
        if args.cvat_img_dir:
            os.makedirs(args.cvat_img_dir, exist_ok=True)

    results = []
    for page_i, f in enumerate(files):
        z = dict(np.load(f))
        W, H = float(z["width"]), float(z["height"])
        coords = z["coords"].reshape(-1, 8).astype(np.float32)
        N = coords.shape[0]
        centers = prim_centers(coords)

        s = args.stride or choose_stride(centers, W, H, args.target, args.margin)
        m = args.margin * s
        T = s + 2 * m
        nx, ny = max(1, math.ceil(W / s)), max(1, math.ceil(H / s))
        tiles = list(make_tiles(centers, W, H, s, m))
        sizes = np.array([len(t[4]) for t in tiles])
        logger.info(f"{osp.basename(f)}: N={N} page={W:.0f}x{H:.0f} stride={s:.1f} tile={T:.1f} "
                    f"grid={nx}x{ny} tiles={len(tiles)} prims/tile median={int(np.median(sizes))} "
                    f"max={sizes.max()}")
        if args.dry_run:
            continue

        page_sem = None
        page_ins = []                                     # (score, label, global_idx)
        t0 = time.time()
        with torch.no_grad():
            for i, j, x0, y0, idx, owned in tqdm.tqdm(tiles, leave=False):
                n = len(idx)
                c, ft, lb, ln, ly = tile_sample(z, idx, x0, y0, T, norm_ds)
                batch = (c, ft, lb, torch.IntTensor([c.shape[0]]), ln, ly)
                torch.cuda.empty_cache()
                with torch.amp.autocast("cuda", enabled=bool(cfg.fp16)):
                    res = model(batch, return_loss=False)

                sem = res["semantic_scores"].float().cpu().numpy()[:n]
                if page_sem is None:
                    page_sem = np.zeros((N, sem.shape[1]), np.float32)
                page_sem[idx[owned]] = sem[owned]         # semantic: core cell owns

                for inst in res["instances"]:
                    score = float(to_np(inst["scores"]))
                    if score < args.keep_thr:
                        continue
                    mk = to_np(inst["masks"]).reshape(-1)[:n]
                    mk = mk if mk.dtype == bool else mk > (0.5 if mk.min() >= 0 and mk.max() <= 1 else 0)
                    if not mk.any():
                        continue
                    gi = idx[mk]
                    pts = coords[gi].reshape(-1, 2)
                    bbox_c = (pts.min(0) + pts.max(0)) / 2
                    if cell_of(bbox_c, s, nx, ny) != (i, j):  # instance owned by another tile
                        continue
                    page_ins.append((score, int(to_np(inst["labels"])), gi))
        logger.info(f"  {len(page_ins)} instances kept, {time.time() - t0:.1f}s")

        instances = []
        for score, label, gi in page_ins:
            mk = np.zeros(N, bool); mk[gi] = True
            instances.append({"masks": mk, "labels": label, "scores": score})

        stem = osp.basename(f).replace("_s2.npz", "")
        pdf_path = osp.abspath(f).replace("_s2.npz", ".pdf")
        results.append({"uuid": stem, "npz_path": osp.abspath(f), "pdf_path": pdf_path,
                        "stride": s, "tile": T, "sem": page_sem, "ins": instances})

        if args.cvat_xml:
            doc = pymupdf.open(pdf_path); page = doc[0]
            img_name = f"{stem}.png"
            if args.cvat_img_dir:
                pix = page.get_pixmap(dpi=args.dpi)
                pix.save(osp.join(args.cvat_img_dir, img_name))
                w, h = pix.width, pix.height
            else:
                sc = args.dpi / 72.0
                w, h = round(page.rect.width * sc), round(page.rect.height * sc)
            sx, sy = w / page.rect.width, h / page.rect.height
            doc.close()
            coords_px = coords.reshape(-1, 4, 2) * np.array([sx, sy], np.float32)
            add_cvat_image(cvat, page_i, img_name, w, h, coords_px, instances,
                           page_sem.argmax(1), num_classes, args.score_thr, args.cvat_mode)

    if args.dry_run:
        return
    np.save(args.out, results, allow_pickle=True)
    logger.info(f"wrote {len(results)} pages to {args.out}")
    if args.cvat_xml:
        write_xml(cvat, args.cvat_xml)
        logger.info(f"wrote {args.cvat_xml}")


if __name__ == "__main__":
    main()
