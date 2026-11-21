import argparse, json, time
from pathlib import Path
from ultralytics import YOLO
from resolve_data import resolve

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--imgsz", type=int, default=320)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--data", default="configs/african_wildlife.yaml")
args = parser.parse_args()

start = time.time()
model = YOLO("yolo11n.pt")
model.train(
    data=resolve(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=16,
    optimizer="SGD", lr0=0.01, seed=0, deterministic=True, workers=2,
    project=str(Path("runs").resolve()), name=args.name, exist_ok=True,
)
minutes = round((time.time() - start) / 60, 1)
json.dump({"name": args.name, "imgsz": args.imgsz, "epochs": args.epochs, "train_minutes": minutes},
          open(f"runs/{args.name}/train_time.json", "w"))
print("training minutes:", minutes)
