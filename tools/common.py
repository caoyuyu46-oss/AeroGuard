import hashlib
import math
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime"))
os.environ.setdefault("WANDB_MODE", "disabled")


def setup_runtime():
    from ultralytics import settings

    keys = ("sync", "hub", "wandb", "mlflow", "comet", "clearml", "neptune", "dvc", "raytune", "tensorboard")
    settings.update({key: False for key in keys if key in settings})


def local_path(value):
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def set_quality_eta(model, eta):
    if eta is None:
        return
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("Quality exponent must be finite and nonnegative")
    layers = getattr(model.model, "model", None)
    head = layers[-1] if layers is not None else None
    if not callable(getattr(head, "set_quality_scoring", None)):
        raise ValueError("Quality scoring can be changed only on a native SJRH checkpoint")
    head.set_quality_scoring(enabled=True, eta=eta)


def resolve_data(config_path, output_dir):
    source = local_path(config_path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    base = local_path(data.get("path", "."))
    data["path"] = str(base)
    for key in ("train", "val", "test"):
        if key not in data:
            continue
        entries = data[key] if isinstance(data[key], list) else [data[key]]
        resolved = [str(Path(x).resolve() if Path(x).is_absolute() else (base / x).resolve()) for x in entries]
        data[key] = resolved if isinstance(data[key], list) else resolved[0]
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256((str(source) + str(base)).encode()).hexdigest()[:12]
    target = target_dir / f"data_{identity}.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target
