import argparse
import math,re
import os,glob,json
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from mmengine.utils import track_parallel_progress
import pymupdf

DEFAULT_LAYER = 0
LABEL_NUM = 35
COMMANDS = ['Line', 'Arc','circle', 'ellipse']
TS = (0.0, 1/3, 2/3, 1.0) # 4-point sampling 


# ===================== BEGIN HELPER FUNCTIONS =====================
def _pt(p):
    return (float(p.x), float(p.y))


def _lerp(a, b, t):
    return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))


def _bezier_pt(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0],
            u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1])


def _bezier_len(p0, p1, p2, p3, n=32):
    prev, total = _bezier_pt(p0, p1, p2, p3, 0.0), 0.0
    for i in range(1, n + 1):
        cur = _bezier_pt(p0, p1, p2, p3, i / n)
        total += math.hypot(cur[0]-prev[0], cur[1]-prev[1])
        prev = cur
    return total
# ====================== END HELPER FUNCTIONS ======================

def parse_pdf(pdf_path, page_idx=0, clip=None):
    doc  = pymupdf.open(pdf_path)
    page = doc[page_idx]
    rect = page.rect if clip is None else pymupdf.Rect(clip)
    W, H = int(rect.width), int(rect.height)

    commands = []
    args = [] # (x1,y1,x2,y2,x3,y3,x4,y4) each primitive smapled at 4 points
    lengths = []
    semanticIds = []
    instanceIds = []
    strokes = []
    layerIds = []
    widths = []
    ocg_to_id = {}

    for d in page.get_drawings():
        if d.get("type") == "f": continue # fill-only

        color = d.get("color") or (0,0,0)
        rgb = [int(round(c * 255)) for c in color]
        w = float(d.get("width") or 1.0)

        lid = d.get("layer") # PDF OCG name, or None
        if lid is None: lid = DEFAULT_LAYER
        else: lid = ocg_to_id.setdefault(lid, len(ocg_to_id))

        for item in d.get("items", []):
            op = item[0]

            if op == "l":
                a, b = _pt(item[1]), _pt(item[2])
                pts  = [_lerp(a, b, t) for t in TS]
                L    = math.hypot(b[0]-a[0], b[1]-a[1])
                cmd  = COMMANDS.index("Line")

            # NOTE: Ellipses and Circles are 4 Bézier curves in a path
            elif op == "c":
                p0, p1 = _pt(item[1]), _pt(item[2])
                p2, p3 = _pt(item[3]), _pt(item[4])
                pts    = [_bezier_pt(p0, p1, p2, p3, t) for t in TS]
                L      = _bezier_len(p0, p1, p2, p3)
                cmd    = COMMANDS.index("Arc")  # closest COMMANDS type

            elif op in ("re", "qu"):
                if op == "re":
                    r = item[1]
                    corners = [(r.x0, r.y0), (r.x1, r.y0),
                               (r.x1, r.y1), (r.x0, r.y1)]
                else:
                    q = item[1]
                    corners = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y),
                               (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
                for i in range(4):  # 4 edges = 4 Line primitives
                    a, b = corners[i], corners[(i+1) % 4]
                    args.append([c for p in [_lerp(a, b, t) for t in TS] for c in p])
                    lengths.append(math.hypot(b[0]-a[0], b[1]-a[1]))
                    commands.append(COMMANDS.index("Line"))
                    semanticIds.append(LABEL_NUM) # background semantic ID
                    instanceIds.append(-1)        # invalid number of instances
                    strokes.append(rgb); widths.append(w); layerIds.append(lid)
                continue

            else:
                continue

            args.append([c for p in pts for c in p])
            lengths.append(L)
            commands.append(cmd)
            semanticIds.append(LABEL_NUM)
            instanceIds.append(-1)
            strokes.append(rgb); 
            widths.append(w); 
            layerIds.append(lid)

    coords = np.asarray(args, dtype=np.float32).reshape(-1, 4, 2) # (N, 4, 2)
    lengths = np.asarray(lengths, dtype=np.float32)     # (N,)
    commands = np.asarray(commands, dtype=np.int16)     # (N,)
    semanticIds = np.asarray(semanticIds, dtype=np.int16)
    instanceIds = np.asarray(instanceIds, dtype=np.int32)
    rgb = np.asarray(strokes, dtype=np.uint8)               # (N, 3)
    widths = np.asarray(widths, dtype=np.float32)
    layerIds = np.asarray(layerIds, dtype=np.int16)

    doc.close()

    return {
        "coords": coords.reshape(len(coords), -1), # (N, 8) float32
        "lengths": lengths,
        "commands": commands,
        "semanticIds": semanticIds,
        "instanceIds": instanceIds,
        "rgb": rgb,
        "widths": widths,
        "layerIds": layerIds,
        "width": np.int32(W),
        "height": np.int32(H)
    }

def process(pdf_path):
    out = parse_pdf(pdf_path)
    out_npz = pdf_path.replace(".pdf","_s2.npz")
    np.savez_compressed(out_npz, **out)

    W, H = int(out["width"]), int(out["height"])
    coords = out["coords"].reshape(-1, 4, 2).astype(np.float32)  # (N, 4, 2)
    coords_norm = coords / np.array([W, H], dtype=np.float32)    # -> [0,1]

    np.savez_compressed(
        pdf_path.replace(".pdf", "_s2_coords.npz"),
        coords=coords_norm,     # (N, 4, 2) float32, normalized
        width=np.int32(W),
        height=np.int32(H),
    )

if __name__=="__main__":
    p = argparse.ArgumentParser(description="Parse Labeled PDFs for Inference")
    p.add_argument("--data", required=True)
    p.add_argument("--nproc", required=True)
    args = p.parse_args()
    pdf_paths = sorted(glob.glob(os.path.join(args.data, '*.pdf')))
    track_parallel_progress(process, pdf_paths, int(args.nproc))
