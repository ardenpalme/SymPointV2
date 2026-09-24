import argparse
import yaml
from munch import Munch
import tqdm
import os
import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time

import torch
import pymupdf

from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.data.svg3 import PDFDataset, SVGDataset, SVG_CATEGORIES
from svgnet.util import get_root_logger, load_checkpoint
from svgnet.evaluation import PointWiseEval, InstanceEval

SKIP = {"wall", "row chairs", "curtain wall"}

# ---------------------------- CVAT helpers ----------------------------
def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def cvat_root(num_classes):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    labels = ET.SubElement(ET.SubElement(ET.SubElement(root, "meta"), "task"), "labels")
    for c in SVG_CATEGORIES[:num_classes]:
        l = ET.SubElement(labels, "label")
        ET.SubElement(l, "name").text = c["name"]
        ET.SubElement(l, "color").text = "#%02x%02x%02x" % tuple(c["color"])
        ET.SubElement(l, "type").text = "any"
        ET.SubElement(l, "attributes")
    return root


def fmt_pts(p):
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in p)


def _bin_mask(m):
    m = np.asarray(m).reshape(-1)
    if m.dtype == bool:
        return m
    # binary 0/1 or sigmoid probs -> 0.5; logits -> 0
    return m > (0.5 if (m.min() >= 0 and m.max() <= 1) else 0)


def resolve_instances(instances, N, score_thr, overlap_thr=0.5):
    """Panoptic merge: highest score first, each primitive owned by at most one instance."""
    items = []
    for inst in instances:
        score = float(to_np(inst["scores"]))
        if score < score_thr:
            continue
        m = _bin_mask(to_np(inst["masks"]))[:N]           # drop padding points
        if m.any():
            items.append((score, int(to_np(inst["labels"])), m))
    items.sort(key=lambda t: -t[0])
    taken = np.zeros(N, dtype=bool)
    out = []
    for score, cls, m in items:
        keep = m & ~taken
        if keep.sum() < overlap_thr * m.sum():            # mostly duplicate of a better instance
            continue
        taken |= keep
        out.append((score, cls, keep))
    return out


def add_cvat_image(root, img_id, name, w, h, coords_px, instances, sem,
                   num_classes, score_thr, mode):
    """coords_px: (N,4,2) primitive points in image pixels.
    sem: per-point semantic argmax (>= N, padded).
    instances: list of {"masks", "labels", "scores"}."""
    img = ET.SubElement(root, "image", id=str(img_id), name=name,
                        width=str(w), height=str(h))
    N = coords_px.shape[0]
    sem = np.asarray(sem).reshape(-1)[:N]
    covered = np.zeros(N, dtype=bool)
    group = 0

    def emit(cls, pts, as_box):
        nonlocal group
        common = dict(label=SVG_CATEGORIES[cls]["name"], source="auto",
                      occluded="0", z_order="0")
        if as_box:
            x0, y0 = pts[..., 0].min(), pts[..., 1].min()
            x1, y1 = pts[..., 0].max(), pts[..., 1].max()
            x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
            ET.SubElement(img, "box", **common,
                          xtl=f"{x0:.2f}", ytl=f"{y0:.2f}",
                          xbr=f"{x1:.2f}", ybr=f"{y1:.2f}")
        else:
            group += 1
            for p in pts:
                ET.SubElement(img, "polyline", **common,
                              points=fmt_pts(p), group_id=str(group))

    # instances, labelled by majority semantic vote (matches the verified overlay)
    for score, cls, m in resolve_instances(instances, N, score_thr):
        votes = sem[m]
        votes = votes[votes < num_classes]
        if votes.size:
            cls = int(np.bincount(votes, minlength=num_classes).argmax())
        if not (0 <= cls < num_classes) or SVG_CATEGORIES[cls]["name"] in SKIP:
            continue
        covered |= m
        emit(cls, coords_px[m], mode == "box" and SVG_CATEGORIES[cls]["isthing"])

    # stuff primitives not claimed by any instance (walls, railings, ...)
    for cls in np.unique(sem[~covered]):
        cls = int(cls)
        if cls >= num_classes or SVG_CATEGORIES[cls]["isthing"] or SVG_CATEGORIES[cls]["name"] in SKIP:
            continue
        emit(cls, coords_px[~covered & (sem == cls)], False)


def write_xml(root, path):
    xml = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
# ----------------------------------------------------------------------


def get_args():
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--datadir", type=str, help="the path to dataset")
    parser.add_argument("--out", type=str, help="the path to save results")
    parser.add_argument("--type", type=str, help="{SVG|PDF}")
    parser.add_argument("--split", type=str, default="val")
    # CVAT export
    parser.add_argument("--cvat_xml", type=str, default=None, help="output annotations.xml")
    parser.add_argument("--cvat_img_dir", type=str, default=None,
                        help="render PDF pages as PNG here (upload these to CVAT)")
    parser.add_argument("--dpi", type=int, default=150, help="render DPI; must match uploaded images")
    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument("--cvat_mode", choices=["box", "polyline"], default="box",
                        help="box: boxes for things, polylines for stuff; polyline: polylines for all")
    return parser.parse_args()


def main():
    args = get_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))
    logger = get_root_logger()

    model = svgnet(cfg.model).cuda()
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)

    if args.type == "SVG":
        ds = SVGDataset(data_root=args.datadir, split=args.split, data_norm="mean",
                        aug=None, repeat=1, logger=logger)
    elif args.type == "PDF":
        ds = PDFDataset(data_root=args.datadir, split=args.split, data_norm="mean",
                        aug=None, repeat=1, logger=logger)
    else:
        logger.error("Invalid dataset type (--type {SVG|PDF})")
        return
    logger.info(f"Load dataset: {len(ds)} files")

    num_classes = cfg.model.semantic_classes
    sem_point_eval = PointWiseEval(num_classes=num_classes, ignore_label=num_classes, gpu_num=1)
    instance_eval = InstanceEval(num_classes=num_classes, ignore_label=num_classes, gpu_num=1)

    datadir_abs = osp.abspath(args.datadir) if args.datadir else None
    out_path = args.out or f"sem_ins_{args.type.lower()}_{args.split}.npy"

    do_cvat = args.cvat_xml is not None and args.type == "PDF"
    if do_cvat:
        cvat = cvat_root(num_classes)
        if args.cvat_img_dir:
            os.makedirs(args.cvat_img_dir, exist_ok=True)

    save_dicts, total_times = [], []
    with torch.no_grad():
        model.eval()
        for i in tqdm.tqdm(range(len(ds))):
            coords, feats, labels, lengths, layerIds = ds[i]
            offset = torch.IntTensor([coords.shape[0]])
            batch = (coords, feats, labels, offset,
                     lengths if lengths is not None else torch.zeros(0), layerIds)

            torch.cuda.empty_cache()
            with torch.amp.autocast('cuda', enabled=cfg.fp16):
                t1 = time.time()
                res = model(batch, return_loss=False)
                total_times.append(time.time() - t1)

            sem_preds = torch.argmax(res["semantic_scores"], dim=1).cpu().numpy()
            sem_gts = res["semantic_labels"].cpu().numpy()
            sem_point_eval.update(sem_preds, sem_gts)
            instance_eval.update(res["instances"], res["targets"], res["lengths"])

            src_path_abs = osp.abspath(ds.data_list[i % len(ds.data_list)])
            stem = osp.basename(src_path_abs).replace("_s2.npz", "").replace("_s2.json", "")
            pdf_path_abs = src_path_abs.replace("_s2.npz", ".pdf").replace("_s2.json", ".pdf")

            if do_cvat:
                # raw primitive coords in PDF points (unnormalized, unpadded)
                coords_pt = np.load(src_path_abs)["coords"].reshape(-1, 4, 2).astype(np.float32)
                doc = pymupdf.open(pdf_path_abs)
                page = doc[0]
                img_name = f"{stem}.png"
                if args.cvat_img_dir:
                    pix = page.get_pixmap(dpi=args.dpi)
                    pix.save(osp.join(args.cvat_img_dir, img_name))
                    w, h = pix.width, pix.height
                else:
                    s = args.dpi / 72.0
                    w, h = round(page.rect.width * s), round(page.rect.height * s)
                sx, sy = w / page.rect.width, h / page.rect.height
                doc.close()
                coords_px = coords_pt * np.array([sx, sy], dtype=np.float32)
                add_cvat_image(cvat, i, img_name, w, h, coords_px, res["instances"],
                               sem_preds, num_classes, args.score_thr, args.cvat_mode)

            save_dicts.append({
                "uuid": stem,
                "datadir": datadir_abs,
                "split": args.split,
                "npz_path": src_path_abs,
                "pdf_path": pdf_path_abs,
                "sem": res["semantic_scores"].cpu().numpy(),
                "ins": res["instances"],
                "targets": res["targets"],
                "lengths": res["lengths"],
            })

    np.save(out_path, save_dicts)
    logger.info(f"Wrote {len(save_dicts)} entries to {out_path}")
    if do_cvat:
        write_xml(cvat, args.cvat_xml)
        logger.info(f"Wrote CVAT 1.1 annotations to {args.cvat_xml}")
    logger.info("Evaluate semantic segmentation")
    sem_point_eval.get_eval(logger)
    logger.info("Evaluate panoptic segmentation")
    instance_eval.get_eval(logger)


if __name__ == "__main__":
    main()
