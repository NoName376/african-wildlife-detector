import sys, random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

root = Path(sys.argv[1])
out = sys.argv[2]
names = ["buffalo", "elephant", "rhino", "zebra"]
colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
imgs = random.Random(0).sample(sorted((root / "images" / "train").glob("*.jpg")), 6)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, p in zip(axes.ravel(), imgs):
    im = Image.open(p)
    w, h = im.size
    ax.imshow(im)
    for row in (root / "labels" / "train" / (p.stem + ".txt")).read_text().split("\n"):
        if not row.strip():
            continue
        c, x, y, bw, bh = map(float, row.split())
        c = int(c)
        ax.add_patch(patches.Rectangle(((x - bw / 2) * w, (y - bh / 2) * h), bw * w, bh * h, fill=False, color=colors[c], lw=2))
        ax.text((x - bw / 2) * w, (y - bh / 2) * h, names[c], color="white", backgroundcolor=colors[c], fontsize=9)
    ax.set_title(p.name, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(out, dpi=80)
