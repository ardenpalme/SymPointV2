"""Usage: PYTHONPATH=./ python check_cvat.py annotations.xml <cvat_img_dir> <outputs.npy>
Writes cvat_render.png (the XML drawn exactly as CVAT should show it) and prints mask stats."""
import sys, os.path as osp
import numpy as np, torch
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from svgnet.data.svg3 import SVG_CATEGORIES
 
xml_path, img_dir, npy_path = sys.argv[1:4]
to_np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
color = {c["name"]: np.array(c["color"]) / 255 for c in SVG_CATEGORIES}
 
# ---- 1. instance mask stats (first entry of the npy) ----
d = np.load(npy_path, allow_pickle=True)[0]
N = np.load(d["npz_path"])["coords"].reshape(-1, 4, 2).shape[0]
print(f"primitives N={N}  sem rows={d['sem'].shape[0]}  instances={len(d['ins'])}")
ins = sorted(d["ins"], key=lambda i: -float(to_np(i["scores"])))
for i in ins[:15]:
    m = to_np(i["masks"]).reshape(-1)
    cls = int(to_np(i["labels"]))
    name = SVG_CATEGORIES[cls]["name"] if 0 <= cls < len(SVG_CATEGORIES) else cls
    print(f"{name:>15} score={float(to_np(i['scores'])):.3f} len={m.size} dtype={m.dtype} "
          f"min={m.min():.3f} max={m.max():.3f} >0:{(m > 0).sum()} >0.5:{(m > 0.5).sum()}")
 
# ---- 2. draw the XML onto the rendered page ----
root = ET.parse(xml_path).getroot()
img_el = root.find("image")
name, W, H = img_el.get("name"), int(img_el.get("width")), int(img_el.get("height"))
img = Image.open(osp.join(img_dir, name))
print(f"\nXML {name}: {W}x{H}   PNG: {img.width}x{img.height}"
      + ("" if (W, H) == img.size else "   <-- SIZE MISMATCH"))
 
plt.figure(figsize=(20, 20 * H / W))
plt.imshow(img, alpha=0.4)
for el in img_el:
    c = color.get(el.get("label"), (1, 0, 0))
    if el.tag == "box":
        x0, y0, x1, y1 = (float(el.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
        plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=c, lw=1.5))
        plt.text(x0, y0, el.get("label"), color=c, fontsize=7)
    elif el.tag == "polyline":
        p = np.array([[float(v) for v in q.split(",")] for q in el.get("points").split(";")])
        plt.plot(p[:, 0], p[:, 1], color=c, lw=0.8)
plt.axis("off")
plt.savefig("cvat_render.png", bbox_inches="tight", dpi=100)
print("wrote cvat_render.png")
