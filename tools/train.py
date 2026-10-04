import argparse
from pathlib import Path

import yaml

from common import ROOT, local_path, resolve_data, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name")
    parser.add_argument("--resume")
    args = parser.parse_args()
    setup_runtime()
    from ultralytics import YOLO

    config = yaml.safe_load(local_path(args.config).read_text(encoding="utf-8"))
    model_config = local_path(config.pop("model"))
    model = YOLO(str(local_path(args.resume))) if args.resume else YOLO(str(model_config))
    if args.weights:
        weights = local_path(args.weights)
        model.load(str(weights) if weights.exists() else args.weights)
    if not args.resume:
        config["pretrained"] = False
    output = local_path(args.project)
    config["data"] = str(resolve_data(args.data, output / "data_configs"))
    config["project"] = str(output)
    config["name"] = args.name or Path(args.config).stem + f"_seed{args.seed if args.seed is not None else config.get('seed', 42)}"
    for key in ("seed", "epochs", "batch", "imgsz", "device", "workers", "amp", "plots"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.resume:
        config["resume"] = True
    model.train(**config)


if __name__ == "__main__":
    main()
