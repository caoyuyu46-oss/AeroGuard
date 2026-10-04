import argparse
import copy
import csv
import json
import math
from collections import defaultdict

import numpy as np

from common import local_path
from metrics import coco_ap, evaluate_at_threshold, match_detections, prepare_predictions, validate_ground_truth


METRICS = ("ap", "recall", "fppi", "bird_overlap_fppi", "few_pixel_recall")
PAPER_SEEDS = (42, 123, 456, 789, 2024)


def resample_groups(ground_truth, predictions_by_model, image_groups, draw):
    images = {image["id"]: image for image in ground_truth["images"]}
    grouped_images, annotations = defaultdict(list), defaultdict(list)
    for image_id in images:
        grouped_images[image_groups[image_id]].append(image_id)
    for annotation in ground_truth["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    grouped_predictions = {}
    for key, predictions in predictions_by_model.items():
        grouped_predictions[key] = defaultdict(list)
        for prediction in predictions:
            grouped_predictions[key][prediction["image_id"]].append(prediction)
    sampled = {"info": copy.deepcopy(ground_truth.get("info", {})),
               "categories": copy.deepcopy(ground_truth["categories"]), "images": [], "annotations": []}
    sampled_predictions = {key: [] for key in predictions_by_model}
    image_id, annotation_id = 0, 0
    for group in draw:
        for original_id in grouped_images[group]:
            image_id += 1
            sampled["images"].append({**images[original_id], "id": image_id})
            for annotation in annotations[original_id]:
                annotation_id += 1
                sampled["annotations"].append({**annotation, "id": annotation_id, "image_id": image_id})
            for key, predictions in grouped_predictions.items():
                for prediction in predictions[original_id]:
                    sampled_predictions[key].append({**prediction, "image_id": image_id})
    return sampled, sampled_predictions


def paired_mean(ground_truth, predictions, thresholds, seeds, drone_id, bird_id):
    results = {}
    for key, values in predictions.items():
        prepared = prepare_predictions(values, ground_truth)
        ap = coco_ap(ground_truth, prepared, [drone_id])["ap"]
        point = evaluate_at_threshold(match_detections(ground_truth, prepared, drone_id, bird_id), thresholds[key])
        results[key] = {"ap": ap, **{metric: point[metric] for metric in METRICS[1:]}}
    differences = {}
    for metric in METRICS:
        values = []
        for seed in seeds:
            candidate, reference = results[("candidate", seed)][metric], results[("reference", seed)][metric]
            if candidate is None or reference is None:
                return None
            values.append(candidate - reference)
        differences[metric] = float(np.mean(values))
    return differences


def run_bootstrap(ground_truth, predictions, thresholds, image_groups, seeds, drone_id, bird_id,
                  replicates=2000, bootstrap_seed=20260928):
    if replicates < 1:
        raise ValueError("At least one replicate is required")
    image_ids = {image["id"] for image in ground_truth["images"]}
    if set(image_groups) != image_ids:
        raise ValueError("The group map must cover every evaluation image exactly once")
    groups = sorted(set(image_groups.values()))
    if not groups:
        raise ValueError("No source groups were supplied")
    expected = {(model, seed) for model in ("reference", "candidate") for seed in seeds}
    if set(predictions) != expected or set(thresholds) != expected:
        raise ValueError("Predictions and validation thresholds must contain both models for every matched seed")
    point = paired_mean(ground_truth, predictions, thresholds, seeds, drone_id, bird_id)
    if point is None:
        raise ValueError("The original dataset must define all five bootstrap metrics")
    rng = np.random.default_rng(bootstrap_seed)
    samples, attempts = [], 0
    while len(samples) < replicates:
        attempts += 1
        if attempts > replicates * 100:
            raise RuntimeError("Too few valid bootstrap samples; check target and group composition")
        draw = [groups[index] for index in rng.integers(0, len(groups), size=len(groups))]
        sampled, sampled_predictions = resample_groups(ground_truth, predictions, image_groups, draw)
        differences = paired_mean(sampled, sampled_predictions, thresholds, seeds, drone_id, bird_id)
        if differences is not None and all(math.isfinite(value) for value in differences.values()):
            samples.append(differences)
    intervals = {metric: {"point_estimate": point[metric],
                          "ci95": np.percentile([sample[metric] for sample in samples], [2.5, 97.5], method="linear").tolist()}
                 for metric in METRICS}
    return {"groups": len(groups), "images": len(image_ids), "matched_seeds": list(seeds),
            "bootstrap_seed": bootstrap_seed, "valid_replicates": len(samples), "attempts": attempts,
            "difference": "candidate minus reference", "thresholds": "Frozen per model and training seed",
            "interval_scope": "Test-source composition conditional on the supplied trained instances",
            "metrics": intervals, "replicates": samples}


def seed_paths(entries):
    result = {}
    for entry in entries:
        seed, path = entry.split("=", 1)
        seed = int(seed)
        if seed in result:
            raise ValueError("Each seed must appear exactly once")
        result[seed] = local_path(path)
    return result


def verified_predictions(path, threshold, seed):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        predictions, metadata = payload["predictions"], payload.get("metadata", {})
    else:
        predictions = payload
        sidecar = path.with_name("prediction_metadata.json")
        if not sidecar.is_file():
            raise ValueError("Bootstrap predictions require wrapper metadata or a prediction_metadata.json sidecar from evaluate.py")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("seed") != seed or not metadata.get("model_id") or metadata.get("model_id") != threshold.get("model_id"):
        raise ValueError("Prediction model identity or seed differs from its frozen threshold")
    if metadata.get("postprocessed") is not True:
        raise ValueError("Bootstrap requires native postprocessed predictions")
    for key, value in (("score_floor", 1e-4), ("max_candidates", 300), ("nms_iou", 0.70)):
        if metadata.get(key) != value or threshold.get(key) != value:
            raise ValueError("Prediction and threshold candidate protocols must match the paper settings")
    if metadata.get("weights_sha256") and threshold.get("weights_sha256") and metadata["weights_sha256"] != threshold["weights_sha256"]:
        raise ValueError("Prediction checkpoint differs from the validation checkpoint")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--reference-thresholds", nargs="+", required=True)
    parser.add_argument("--candidate-thresholds", nargs="+", required=True)
    parser.add_argument("--expected-groups", type=int, default=312)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260928)
    parser.add_argument("--drone-category", type=int, default=2)
    parser.add_argument("--bird-category", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ground_truth = validate_ground_truth(json.loads(local_path(args.annotations).read_text(encoding="utf-8")))
    image_ids = {str(image["id"]): image["id"] for image in ground_truth["images"]}
    groups = {}
    with local_path(args.groups).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] not in image_ids or image_ids[row["image_id"]] in groups or not row["group_id"]:
                raise ValueError("Group CSV requires unique known image_id values and nonempty group_id values")
            groups[image_ids[row["image_id"]]] = row["group_id"]
    if len(set(groups.values())) != args.expected_groups:
        raise ValueError("The supplied source-group count differs from --expected-groups")
    predictions, thresholds = {}, {}
    for model in ("reference", "candidate"):
        paths = seed_paths(getattr(args, model))
        threshold_paths = seed_paths(getattr(args, model + "_thresholds"))
        if set(paths) != set(PAPER_SEEDS) or set(threshold_paths) != set(PAPER_SEEDS):
            raise ValueError("Supply all five matched paper seeds: 42, 123, 456, 789, 2024")
        for seed in PAPER_SEEDS:
            threshold = json.loads(threshold_paths[seed].read_text(encoding="utf-8"))
            if threshold.get("selection_split") != "val":
                raise ValueError("Bootstrap requires validation-selected thresholds")
            if threshold.get("seed") is not None and int(threshold["seed"]) != seed:
                raise ValueError("Threshold seed differs from its assigned training seed")
            if threshold.get("drone_category_id") != args.drone_category or threshold.get("bird_category_id") != args.bird_category:
                raise ValueError("Threshold category identifiers differ from bootstrap categories")
            predictions[(model, seed)] = verified_predictions(paths[seed], threshold, seed)
            thresholds[(model, seed)] = float(threshold["threshold"])
    report = run_bootstrap(ground_truth, predictions, thresholds, groups, PAPER_SEEDS, args.drone_category,
                           args.bird_category, args.replicates, args.bootstrap_seed)
    destination = local_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
