import argparse, os, os.path as osp, shutil, glob
import numpy as np
import pymupdf
from tqdm import tqdm

# ------- turn one instance dict into (point_indices, sem_id, score) -------
def unpack_instance(inst):
    mask = inst["masks"]
    if hasattr(mask, "cpu"):                 # in case it's still a tensor
        mask = mask.cpu().numpy()
    mask = np.asarray(mask).astype(bool)     # (N,)
    pinds = np.nonzero(mask)[0]
    return pinds, int(inst["labels"]), float(inst["scores"])


def bbox(coords, pinds):
    # coords: (N, 4, 2) in PDF points
    pts = coords[pinds].reshape(-1, 2)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    return float(x0), float(y0), float(x1), float(y1)


def draw(page, bb, text, fs=8):
    x0, y0, x1, y1 = bb
    page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(1, 0, 0), width=1.2)
    tw = pymupdf.get_text_length(text, fontsize=fs)
    ly = max(0.0, y0 - fs - 2)
    page.draw_rect(pymupdf.Rect(x0, ly, x0 + tw + 2, ly + fs + 2),
                   color=(1, 0, 0), fill=(1, 1, 1), width=0.4)
    page.insert_text((x0 + 1, ly + fs), text, fontsize=fs, color=(1, 0, 0))


def find_pdf(entry, pdf_root):
    p = entry.get("pdf_path")
    if p and osp.exists(p):
        return p
    uuid = entry.get("uuid")
    hits = glob.glob(osp.join(pdf_root, "**", uuid + ".pdf"), recursive=True)
    if len(hits) == 1: return hits[0]
    if len(hits) > 1:  raise RuntimeError(f"ambiguous uuid {uuid}: {hits}")
    return None


def find_map(entry, pdf_root, mapdir):
    uuid = entry.get("uuid")
    root = mapdir or pdf_root
    hits = glob.glob(osp.join(root, "**", f"{uuid}_s2_coords.npz"), recursive=True)
    if len(hits) == 1: return hits[0]
    if len(hits) > 1:  raise RuntimeError(f"ambiguous map for {uuid}: {hits}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--pdf_root", required=True)
    ap.add_argument("--mapdir", default=None)
    ap.add_argument("--out", default="vis_out")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--min_score", type=float, default=0.1)  # same as InstanceEval
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for entry in tqdm(np.load(args.results, allow_pickle=True), desc="annotating"):
        pdf = find_pdf(entry, args.pdf_root)
        mp  = find_map(entry, args.pdf_root, args.mapdir)
        if pdf is None or mp is None:
            print("skip", entry.get("uuid")); continue

        m = np.load(mp)
        W, H = int(m["width"]), int(m["height"])
        coords = m["coords"] * np.array([W, H], dtype=np.float32)   # (N,4,2)

        # mirror InstanceEval: filter + sort by score desc
        insts = [i for i in entry["ins"] if float(i["scores"]) >= args.min_score]
        insts.sort(key=lambda x: float(x["scores"]), reverse=True)

        # copy -> annotate copy
        out_pdf = osp.join(args.out, f"{entry['uuid']}.pdf")
        shutil.copy2(pdf, out_pdf)
        doc  = pymupdf.open(out_pdf)
        page = doc[args.page]

        for rank, inst in enumerate(insts):
            pinds, sem_id, score = unpack_instance(inst)
            if pinds.size == 0:
                continue
            # no explicit instance id in the schema -> use rank, or score
            draw(page, bbox(coords, pinds),
                 f"sem={sem_id} ins={rank} s={score:.2f}")

        doc.saveIncr()
        doc.close()

    print("done ->", args.out)


if __name__ == "__main__":
    main()
