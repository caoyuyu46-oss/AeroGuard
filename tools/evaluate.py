import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import ROOT, local_path, set_quality_eta, setup_runtime
from tools.metrics import (
    MAX_CANDIDATES,
    NMS_IOU,
    SCORE_FLOOR,
    coco_ap,
    evaluate_at_threshold,
    match_detections,
    operating_curve,
    prepare_predictions,
    select_threshold,
    sweep_operating_points,
    validate_ground_truth,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def data_ground_truth(data_path, split, annotations_path=None, image_root=None):
    config = yaml.safe_load(local_path(data_path).read_text(encoding="utf-8")) if data_path else {}
    base = local_path(config.get("path", ".")) if data_path else None
    annotation_value = annotations_path or config.get("annotations", {}).get(split)
    if annotation_value:
        annotation_file = local_path(annotation_value) if annotations_path else Path(annotation_value).expanduser()
        if not annotation_file.is_absolute():
            annotation_file = (base / annotation_file).resolve()
        data = validate_ground_truth(json.loads(annotation_file.read_text(encoding="utf-8")))
        image_base = local_path(image_root) if image_root else base or annotation_file.parent
        for image in data["images"]:
            if "file_name" in image:
                path = Path(image["file_name"])
                if not path.is_absolute():
                    choices = [image_base / path, image_base / "images" / split / path, annotation_file.parent / path]
                    path = next((candidate for candidate in choices if candidate.exists()), choices[0])
                image["_file_path"] = str(path.resolve())
        return data, {"source": str(annotation_file), "format": "COCO", "crowd_ignore_from_source": True}
    if not data_path or split not in config:
        raise ValueError("Supply --annotations or a data YAML containing the requested split")
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or sorted(int(key) for key in names) != list(range(len(names))):
        raise ValueError("YOLO data names must use contiguous zero-based class indices")
    categories = [{"id": int(index) + 1, "name": name} for index, name in names.items()]
    image_paths = []
    entries = config[split] if isinstance(config[split], list) else [config[split]]
    extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    for entry in entries:
        path = Path(entry).expanduser()
        path = path if path.is_absolute() else base / path
        if path.is_dir():
            image_paths.extend(candidate.resolve() for candidate in path.rglob("*") if candidate.suffix.lower() in extensions)
        elif path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                candidate = Path(line.strip())
                if not candidate.is_absolute():
                    candidate = path.parent / candidate if line.strip().startswith("./") else base / candidate
                image_paths.append(candidate.resolve())
        else:
            image_paths.append(path.resolve())
    images, annotations = [], []
    for image_id, path in enumerate(sorted(set(image_paths), key=lambda value: value.as_posix()), 1):
        with Image.open(path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": str(path), "width": width, "height": height, "_file_path": str(path)})
        parts = list(path.parts)
        image_positions = [index for index, part in enumerate(parts) if part == "images"]
        if image_positions:
            parts[image_positions[-1]] = "labels"
            label = Path(*parts).with_suffix(".txt")
        else:
            label = path.with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Missing label file: {label}; explicit empty files are required for background images")
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Expected five detection-label fields in {label}")
            category, center_x, center_y, box_width, box_height = map(float, fields)
            if not all(math.isfinite(value) for value in (category, center_x, center_y, box_width, box_height)) or category != int(category):
                raise ValueError(f"Invalid class or coordinates in {label}")
            x, y = (center_x - box_width / 2) * width, (center_y - box_height / 2) * height
            w, h = box_width * width, box_height * height
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": int(category) + 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    return validate_ground_truth({"images": images, "annotations": annotations, "categories": categories}), {
        "source": str(local_path(data_path)), "format": "YOLO detection labels", "crowd_ignore_from_source": False,
    }


def category_id(data, requested, names, required=True):
    known = {item["id"] for item in data["categories"]}
    if requested is not None:
        if requested not in known:
            raise ValueError(f"Unknown COCO category identifier: {requested}")
        return requested
    matches = [item["id"] for item in data["categories"] if str(item.get("name", "")).strip().lower() in names]
    if len(matches) == 1:
        return matches[0]
    if required:
        raise ValueError("Specify --drone-category-id when drone/uav cannot be identified unambiguously by name")
    return None


def weight_predictions(args, ground_truth):
    weights = local_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights must be an existing local file: {weights}")
    setup_runtime()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    set_quality_eta(model, args.quality_eta)
    target_names = {str(category.get("name", "")).lower(): category["id"] for category in ground_truth["categories"]}
    mapping = {}
    for index, name in model.names.items():
        normalized = str(name).lower()
        if normalized in ("uav", "drone"):
            mapping[index] = target_names.get("drone", target_names.get("uav"))
        else:
            mapping[index] = target_names.get(normalized)
    if not any(value is not None for value in mapping.values()):
        raise ValueError("No model class names match the ground-truth category names")
    images = []
    lookup = {}
    for image in ground_truth["images"]:
        path = Path(image.get("_file_path", ""))
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation image is unavailable: {image.get('file_name', image['id'])}")
        images.append(str(path.resolve()))
        lookup[str(path.resolve())] = image
    predictions = []
    results = model.predict(source=images, stream=True, imgsz=args.imgsz, conf=SCORE_FLOOR, iou=NMS_IOU, max_det=MAX_CANDIDATES, agnostic_nms=False, device=args.device, save=False, verbose=False)
    for result in results:
        image = lookup[str(Path(result.path).resolve())]
        if tuple(result.orig_shape) != (image["height"], image["width"]):
            raise ValueError("Model input dimensions do not match the original-image ground truth")
        for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            target_id = mapping.get(int(cls))
            if target_id is None:
                continue
            predictions.append({"image_id": image["id"], "category_id": target_id, "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "score": score})
    metadata = {"source": str(weights), "postprocessed": True, "class_mapping": mapping, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "native_postprocessing": "detector_native", "class_aware_nms": None}
    if not getattr(model.model.model[-1], "end2end", False):
        metadata.update(native_postprocessing="class_aware_nms", class_aware_nms=True)
    if args.quality_eta is not None:
        metadata["quality_eta"] = args.quality_eta
    return predictions, metadata


def build_parser():
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--weights")
    inputs.add_argument("--predictions")
    parser.add_argument("--data")
    parser.add_argument("--annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--quality-eta", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-from")
    parser.add_argument("--fppi-budget", type=float, default=0.05)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--recall-target", type=float, default=0.90)
    parser.add_argument("--drone-category-id", type=int)
    parser.add_argument("--bird-category-id", type=int)
    parser.add_argument("--ap-only", action="store_true")
    parser.add_argument("--ap-category-ids", type=int, nargs="+")
    parser.add_argument("--raw-predictions", action="store_true")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    return parser


def run(args):
    if args.quality_eta is not None and (not args.weights or not math.isfinite(args.quality_eta) or args.quality_eta < 0):
        raise ValueError("--quality-eta requires weights and a finite nonnegative exponent")
    if args.threshold_from and (args.split != "test" or args.ap_only):
        raise ValueError("Frozen thresholds apply to test FROC evaluation only")
    if args.weights and args.raw_predictions:
        raise ValueError("Weight-based inference already applies detector postprocessing")
    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ground_truth, gt_metadata = data_ground_truth(args.data, args.split, args.annotations, args.image_root)
    write_json(output / "ground_truth.json", ground_truth)
    if args.predictions:
        source = local_path(args.predictions)
        payload = json.loads(source.read_text(encoding="utf-8"))
        predictions = payload if isinstance(payload, list) else payload["predictions"]
        prediction_metadata = {} if isinstance(payload, list) else payload.get("metadata", {})
        sidecar = source.with_name("prediction_metadata.json")
        if isinstance(payload, list) and source.name == "predictions.json" and sidecar.is_file():
            prediction_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if "seed" in prediction_metadata and prediction_metadata["seed"] != args.seed:
            raise ValueError("Prediction metadata seed differs from --seed")
        if prediction_metadata.get("postprocessed") is False and not args.raw_predictions:
            raise ValueError("Raw predictions require --raw-predictions")
        prediction_metadata = {**prediction_metadata, "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    else:
        predictions, prediction_metadata = weight_predictions(args, ground_truth)
        if args.quality_eta is not None:
            prediction_metadata["quality_eta"] = args.quality_eta
    metadata_model = prediction_metadata.get("model_id")
    if args.model_id is not None and metadata_model is not None and args.model_id != metadata_model:
        raise ValueError("Prediction metadata model identifier differs from --model-id")
    automatic_model = prediction_metadata.get("weights_sha256")
    if automatic_model and args.quality_eta is not None:
        automatic_model += "|quality_eta=" + format(args.quality_eta, ".17g")
    model_id = args.model_id or metadata_model or automatic_model
    if args.predictions and not args.ap_only and (args.split == "val" or args.threshold_from) and not model_id:
        raise ValueError("Offline threshold selection and frozen evaluation require --model-id or metadata.model_id")
    prepared = prepare_predictions(predictions, ground_truth, raw_predictions=args.raw_predictions)
    write_json(output / "predictions.json", [{key: item[key] for key in ("image_id", "category_id", "bbox", "score")} for item in prepared])
    drone = category_id(ground_truth, args.drone_category_id, {"drone", "uav"}) if not args.ap_only else None
    bird = category_id(ground_truth, args.bird_category_id, {"bird"}, required=False) if not args.ap_only else None
    prediction_metadata.update(seed=args.seed, model_id=model_id, postprocessed=True, score_floor=SCORE_FLOOR, max_candidates=MAX_CANDIDATES, nms_iou=NMS_IOU, drone_category_id=drone, bird_category_id=bird)
    if args.raw_predictions:
        prediction_metadata.update(native_postprocessing="class_aware_nms", class_aware_nms=True)
    else:
        prediction_metadata.setdefault("native_postprocessing", "supplied_postprocessed")
        prediction_metadata.setdefault("class_aware_nms", None)
    write_json(output / "prediction_metadata.json", prediction_metadata)
    ap_categories = args.ap_category_ids or ([drone] if drone is not None else None)
    report = {
        "split": args.split,
        "seed": args.seed,
        "model_id": model_id,
        "ground_truth": gt_metadata,
        "predictions": prediction_metadata,
        "image_count": len(ground_truth["images"]),
        "prepared_prediction_count": len(prepared),
        "protocol": {"score_floor": SCORE_FLOOR, "nms_iou": NMS_IOU, "froc_cross_class_max_candidates": MAX_CANDIDATES, "input_postprocessed": not args.raw_predictions, "stable_indices": "input prediction order and ground-truth annotation order", "category_schema": "COCO category identifiers; YOLO labels are converted with category_id=class_index+1"},
        "coco": coco_ap(ground_truth, prepared, ap_categories),
    }
    if not args.ap_only:
        matching = match_detections(ground_truth, prepared, drone, bird)
        sweep = sweep_operating_points(matching, args.budgets, args.recall_target)
        curve = sweep.pop("curve")
        report["counts"] = {key: value for key, value in matching.items() if key != "records"}
        report["score_sweep"] = {"selection_set": args.split, **sweep}
        with (output / "froc_curve.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        write_json(output / "matched_detections.json", matching)
        if args.split == "val":
            selected = select_threshold(curve, args.fppi_budget)
            threshold = {"threshold": selected["threshold"], "selection_split": "val", "seed": args.seed, "model_id": model_id, "weights_sha256": prediction_metadata.get("weights_sha256"), "fppi_budget": args.fppi_budget, "drone_category_id": drone, "bird_category_id": bird, "selection_metrics": selected, "score_floor": SCORE_FLOOR, "max_candidates": MAX_CANDIDATES, "nms_iou": NMS_IOU}
            threshold.update(quality_eta=prediction_metadata.get("quality_eta"), native_postprocessing=prediction_metadata["native_postprocessing"], class_aware_nms=prediction_metadata["class_aware_nms"])
            write_json(output / "threshold.json", threshold)
            report["validation_selected"] = threshold
        if args.threshold_from:
            frozen = json.loads(local_path(args.threshold_from).read_text(encoding="utf-8"))
            if frozen.get("selection_split") != "val" or frozen.get("seed") != args.seed:
                raise ValueError("Frozen threshold must come from validation for the same seed")
            if frozen.get("model_id") != model_id:
                raise ValueError("Frozen threshold model identifier differs from the current model")
            if frozen.get("quality_eta") != prediction_metadata.get("quality_eta"):
                raise ValueError("Frozen threshold quality exponent differs from the current scoring configuration")
            if frozen.get("drone_category_id") != drone or frozen.get("bird_category_id") != bird:
                raise ValueError("Frozen threshold category identifiers differ from the current dataset")
            if frozen.get("weights_sha256") is not None and prediction_metadata.get("weights_sha256") is not None and frozen["weights_sha256"] != prediction_metadata["weights_sha256"]:
                raise ValueError("Frozen threshold checkpoint differs from the current checkpoint")
            for key, value in (("score_floor", SCORE_FLOOR), ("max_candidates", MAX_CANDIDATES), ("nms_iou", NMS_IOU)):
                if frozen.get(key) != value:
                    raise ValueError("Frozen threshold uses a different candidate protocol")
            report["frozen_validation_threshold"] = {"threshold_source": str(local_path(args.threshold_from)), "selection_fppi_budget": frozen["fppi_budget"], **evaluate_at_threshold(matching, frozen["threshold"])}
    write_json(output / "metrics.json", report)
    return report


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps({"output": result["split"], "seed": result["seed"], "coco": result["coco"]}, indent=2, allow_nan=False))
