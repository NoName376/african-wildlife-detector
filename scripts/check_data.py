import hashlib, sys
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import imagehash

root = Path(sys.argv[1])
names = ["buffalo", "elephant", "rhino", "zebra"]
hashes = {}
for split in ["train", "val", "test"]:
    imgs = sorted((root / "images" / split).glob("*"))
    objs, empty, bad = Counter(), [], []
    for p in imgs:
        lab = root / "labels" / split / (p.stem + ".txt")
        rows = [r.split() for r in lab.read_text().splitlines() if r.strip()] if lab.exists() else []
        if not rows:
            empty.append(p.name)
        for r in rows:
            c, *box = r
            box = list(map(float, box))
            if int(c) not in range(4) or len(box) != 4 or any(v < 0 or v > 1 for v in box) or box[2] <= 0 or box[3] <= 0:
                bad.append((p.name, r))
            objs[names[int(c)]] += 1
        im = Image.open(p)
        hashes[(split, p.name)] = (hashlib.md5(p.read_bytes()).hexdigest(), imagehash.phash(im))
    print(split, "images", len(imgs), "objects", sum(objs.values()), dict(objs), "empty", len(empty), "bad", len(bad))
    if bad: print(" bad:", bad[:5])
    if empty: print(" empty:", empty[:5])

exact = defaultdict(list)
for k, (md5, _) in hashes.items():
    exact[md5].append(k)
cross = [v for v in exact.values() if len({s for s, _ in v}) > 1]
print("exact duplicate groups:", sum(len(v) > 1 for v in exact.values()), "across splits:", len(cross))
items = list(hashes.items())
near = []
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        (a, (_, ha)), (b, (_, hb)) = items[i], items[j]
        if a[0] != b[0] and ha - hb <= 4:
            near.append((a, b, ha - hb))
print("near-duplicate pairs across splits:", len(near))
for n in near[:30]: print(" ", n)
