import sys
from pathlib import Path

root = Path(sys.argv[1]) / "labels"
fixes = {
    "test/val_3_376.txt": (3, 2),
    "train/train_3_373.txt": (3, 2),
    "train/train_3_374.txt": (3, 2),
    "train/train_3_375.txt": (3, 2),
    "train/train_4_376.txt": (0, 3),
}
for name, (old, new) in fixes.items():
    f = root / name
    rows = [r.split() for r in f.read_text().splitlines() if r.strip()]
    rows = [[str(new) if r[0] == str(old) else r[0]] + r[1:] for r in rows]
    f.write_text("\n".join(" ".join(r) for r in rows) + "\n")
    print("fixed", name, old, "->", new)
