from pathlib import Path
import yaml

def resolve(cfg="configs/african_wildlife.yaml"):
    data = yaml.safe_load(open(cfg))
    data["path"] = str(Path(data["path"]).resolve())
    Path("runs").mkdir(exist_ok=True)
    yaml.safe_dump(data, open("runs/data.yaml", "w"))
    return "runs/data.yaml"
