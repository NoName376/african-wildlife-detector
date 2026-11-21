import random, shutil, sys
from pathlib import Path
from PIL import Image
import imagehash

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
SEED, THRESHOLD = 0, 10

images = sorted(p for p in src.glob("images/*/*") if p.suffix.lower() == ".jpg")
hashes = [imagehash.phash(Image.open(p)) for p in images]
parent = list(range(len(images)))

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

for i in range(len(images)):
    for j in range(i + 1, len(images)):
        if hashes[i] - hashes[j] <= THRESHOLD:
            parent[find(i)] = find(j)

groups = {}
for i, p in enumerate(images):
    groups.setdefault(find(i), []).append(p)
groups = list(groups.values())
random.Random(SEED).shuffle(groups)

n, split_of, count = len(images), {}, 0
for g in groups:
    split = "train" if count < 0.70 * n else "val" if count < 0.85 * n else "test"
    for p in g:
        split_of[p] = split
    count += len(g)

for p, split in split_of.items():
    new_name = f"{p.parent.name}_{p.name}".replace(" ", "_").replace("(", "").replace(")", "")
    for kind, ext, f in [("images", ".jpg", p), ("labels", ".txt", src / "labels" / p.parent.name / (p.stem + ".txt"))]:
        (dst / kind / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst / kind / split / (Path(new_name).stem + ext))

print(len(images), "images,", len(groups), "groups,", sum(len(g) > 1 for g in groups), "groups with duplicates")
for s in ["train", "val", "test"]:
    print(s, sum(v == s for v in split_of.values()))
