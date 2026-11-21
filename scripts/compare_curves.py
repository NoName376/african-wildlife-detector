import pandas as pd
import matplotlib.pyplot as plt

runs = {"baseline_320": "imgsz 320", "imgsz_416": "imgsz 416"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for run, label in runs.items():
    df = pd.read_csv(f"results/{run}/results.csv")
    axes[0].plot(df["epoch"], df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"], label=label)
    axes[1].plot(df["epoch"], df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"], label=label)
    axes[2].plot(df["epoch"], df["metrics/mAP50(B)"], label=label)
for ax, t in zip(axes, ["train loss (box+cls+dfl)", "val loss (box+cls+dfl)", "val mAP@0.50"]):
    ax.set_title(t)
    ax.set_xlabel("epoch")
    ax.legend()
plt.tight_layout()
plt.savefig("results/learning_curves_comparison.png", dpi=80)
for run in runs:
    df = pd.read_csv(f"results/{run}/results.csv")
    i = df["metrics/mAP50(B)"].idxmax()
    print(run, "best mAP50 epoch", int(df.loc[i, "epoch"]), round(df.loc[i, "metrics/mAP50(B)"], 3), "last", round(df["metrics/mAP50(B)"].iloc[-1], 3))
