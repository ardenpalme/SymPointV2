import numpy as np, pymupdf, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from svgnet.data.svg3 import SVG_CATEGORIES

d = np.load("outputs.npy", allow_pickle=True)[0]          # the PDF-run output
coords = np.load(d["npz_path"])["coords"].reshape(-1, 4, 2)
sem = d["sem"].argmax(1)
print("coords", len(coords), "preds", len(sem))           # should match
sem = sem[:len(coords)]

page = pymupdf.open(d["pdf_path"])[0]; dpi = 100; s = dpi / 72
pix = page.get_pixmap(dpi=dpi)
img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
plt.figure(figsize=(20, 20)); plt.imshow(img, alpha=0.4)
for p, c in zip(coords, sem):
    if c < 35:
        plt.plot(p[:, 0] * s, p[:, 1] * s, color=np.array(SVG_CATEGORIES[c]["color"]) / 255, lw=1)
plt.axis("off"); plt.savefig("overlay.png", bbox_inches="tight")
