import argparse, json
from pathlib import Path
from ultralytics import YOLO
from resolve_data import resolve

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", default=["baseline_320", "imgsz_416"])
parser.add_argument("--split", default="val")
args = parser.parse_args()

data = resolve()
results = {}
for run in args.runs:
    train_info = json.load(open(f"runs/{run}/train_time.json"))
    m = YOLO(f"runs/{run}/weights/best.pt").val(
        data=data, split=args.split, imgsz=train_info["imgsz"], batch=16, conf=0.001, iou=0.7,
        project=str(Path("runs").resolve()), name=f"eval_{run}_{args.split}", exist_ok=True, plots=True,
    )
    results[run] = {
        "imgsz": train_info["imgsz"], "train_minutes": train_info["train_minutes"],
        "precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
        "mAP50": round(float(m.box.map50), 3), "mAP50_95": round(float(m.box.map), 3),
        "per_class_mAP50": {m.names[i]: round(float(v), 3) for i, v in zip(m.box.ap_class_index, m.box.ap50)},
    }
json.dump(results, open(f"results/metrics_{args.split}.json", "w"), indent=2)
for run, r in results.items():
    print(run, r)
