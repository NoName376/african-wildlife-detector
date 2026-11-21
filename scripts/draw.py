import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

split = sys.argv[1]
report = {x["image"]: x for x in json.load(open(f"results/{split}_predictions.json"))}
names = ["buffalo", "elephant", "rhino", "zebra"]
colors = dict(zip(names, ["tab:orange", "tab:blue", "tab:green", "tab:red"]))
out, files = sys.argv[2], sys.argv[3:]
root = Path("datasets/aw-split")
cols = 3
rows = (len(files) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 4.8 * rows), squeeze=False)
for ax, f in zip(axes.ravel(), files):
    im = Image.open(root / "images" / split / f)
    w, h = im.size
    ax.imshow(im)
    for line in (root / "labels" / split / (Path(f).stem + ".txt")).read_text().splitlines():
        c, x, y, bw, bh = map(float, line.split())
        ax.add_patch(patches.Rectangle(((x - bw / 2) * w, (y - bh / 2) * h), bw * w, bh * h, fill=False, ls="--", color="white", lw=1.5))
    for name, (x1, y1, x2, y2), s in report[f]["boxes"]:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=colors[name], lw=2.5))
        ax.text(x1, y1, f"{name} {s:.2f}", color="white", backgroundcolor=colors[name], fontsize=9)
    ax.set_title(f, fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
fig.suptitle("colored = prediction (class, confidence), white dashed = ground truth", fontsize=11)
plt.tight_layout()
plt.savefig(out, dpi=80)
