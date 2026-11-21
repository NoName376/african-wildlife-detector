import argparse, json
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--split", default="test")
args = parser.parse_args()

imgsz = json.load(open(f"runs/{args.run}/train_time.json"))["imgsz"]
model = YOLO(f"runs/{args.run}/weights/best.pt")
names = model.names
root = Path("datasets/aw-split")
images = sorted((root / "images" / args.split).glob("*.jpg"))

def load_gt(p, w, h):
    rows = [list(map(float, r.split())) for r in (root / "labels" / args.split / (p.stem + ".txt")).read_text().splitlines() if r.strip()]
    return [(int(c), [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]) for c, x, y, bw, bh in rows]

report = []
for p in images:
    r = model.predict(str(p), imgsz=imgsz, conf=args.conf, verbose=False)[0]
    h, w = r.orig_shape
    gt = load_gt(p, w, h)
    pred = [(int(c), b.tolist(), float(s)) for c, b, s in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf)]
    used, missed, wrong_class, false_pos = set(), [], [], []
    for gc, gb in gt:
        best, best_iou = None, 0.5
        for i, (pc, pb, ps) in enumerate(pred):
            iou = box_iou(torch.tensor([gb]), torch.tensor([pb]))[0, 0].item()
            if i not in used and iou >= best_iou:
                best, best_iou = i, iou
        if best is None:
            missed.append(names[gc])
        else:
            used.add(best)
            if pred[best][0] != gc:
                wrong_class.append(f"{names[gc]} -> {names[pred[best][0]]}")
    false_pos = [names[pc] for i, (pc, pb, ps) in enumerate(pred) if i not in used]
    report.append({"image": p.name, "gt": len(gt), "pred": len(pred), "missed": missed, "wrong_class": wrong_class, "false_pos": false_pos,
                   "boxes": [(names[c], [round(v) for v in b], round(s, 2)) for c, b, s in pred]})

json.dump(report, open(f"results/{args.split}_predictions.json", "w"), indent=1)
errors = [x for x in report if x["missed"] or x["wrong_class"] or x["false_pos"]]
summary = {
    "images": len(report), "images_with_errors": len(errors),
    "missed": sum(len(x["missed"]) for x in report), "wrong_class": sum(len(x["wrong_class"]) for x in report),
    "false_pos": sum(len(x["false_pos"]) for x in report), "conf": args.conf, "match_iou": 0.5,
}
json.dump(summary, open(f"results/{args.split}_error_summary.json", "w"), indent=2)
print(summary)
for x in errors:
    print(x["image"], "missed", x["missed"], "wrong", x["wrong_class"], "fp", x["false_pos"])
