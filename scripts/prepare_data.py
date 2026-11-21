import subprocess, sys, urllib.request, zipfile
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
raw, split = Path("datasets/african-wildlife"), Path("datasets/aw-split")
if not raw.exists():
    Path("datasets").mkdir(exist_ok=True)
    urllib.request.urlretrieve(URL, "datasets/african-wildlife.zip")
    zipfile.ZipFile("datasets/african-wildlife.zip").extractall(raw)
if not split.exists():
    subprocess.run([sys.executable, "scripts/resplit.py", str(raw), str(split)], check=True)
    subprocess.run([sys.executable, "scripts/fix_labels.py", str(split)], check=True)
print("data ready in", split)
