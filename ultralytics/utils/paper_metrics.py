import contextlib
import copy
import io
import math

import numpy as np


SCORE_FLOOR = 1e-4
NMS_IOU = 0.70
MATCH_IOU = 0.50
MAX_CANDIDATES = 300
REJECT_ALL_THRESHOLD = math.nextafter(1.0, math.inf)


def validate_ground_truth(ground_truth):
    data = copy.deepcopy(ground_truth)
    for key in ("images", "annotations", "categories"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"COCO ground truth requires a {key} list")
    images = {}
    for image in data["images"]:
        image_id = image["id"]
        if image_id in images or image["width"] <= 0 or image["height"] <= 0:
            raise ValueError("Image identifiers must be unique and dimensions must be positive")
        images[image_id] = image
    if not images:
        raise ValueError("At least one evaluation image is required")
    categories = {category["id"] for category in data["categories"]}
    if len(categories) != len(data["categories"]):
        raise ValueError("Category identifiers must be unique")
    annotation_ids = set()
    for index, annotation in enumerate(data["annotations"]):
        annotation.setdefault("id", index + 1)
        if annotation["id"] in annotation_ids:
            raise ValueError("Annotation identifiers must be unique")
        annotation_ids.add(annotation["id"])
        if annotation["image_id"] not in images or annotation["category_id"] not in categories:
            raise ValueError("Annotations must reference known images and categories")
        box = annotation["bbox"]
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box) or min(box[2:]) <= 0:
            raise ValueError("Ground-truth boxes must be finite COCO xywh boxes with positive size")
        annotation.setdefault("area", float(box[2] * box[3]))
        annotation.setdefault("iscrowd", 0)
    data.setdefault("info", {})
    return data


def box_iou_xywh(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    left = np.maximum(box[:2], boxes[:, :2])
    right = np.minimum(box[:2] + box[2:], boxes[:, :2] + boxes[:, 2:])
    intersection = np.maximum(right - left, 0).prod(axis=1)
    union = box[2:].prod() + boxes[:, 2:].prod(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def prepare_predictions(predictions, ground_truth, raw_predictions=False, score_floor=SCORE_FLOOR, max_candidates=MAX_CANDIDATES):
    image_ids = {image["id"] for image in ground_truth["images"]}
    category_ids = {category["id"] for category in ground_truth["categories"]}
    groups = {image_id: [] for image_id in image_ids}
    for index, prediction in enumerate(predictions):
        item = dict(prediction)
        if item["image_id"] not in image_ids or item["category_id"] not in category_ids:
            raise ValueError("Predictions must reference known images and COCO category identifiers")
        score = float(item["score"])
        box = [float(value) for value in item["bbox"]]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction scores must be finite values in [0, 1]")
        if len(box) != 4 or not all(math.isfinite(value) for value in box) or min(box[2:]) < 0:
            raise ValueError("Predictions require finite COCO xywh boxes with nonnegative size")
        if score < score_floor:
            continue
        item.update(score=score, bbox=box, _index=index)
        groups[item["image_id"]].append(item)
    result = []
    for image in ground_truth["images"]:
        ordered = sorted(groups[image["id"]], key=lambda item: (-item["score"], item["_index"]))
        if raw_predictions:
            kept = []
            for item in ordered:
                same_class = [other["bbox"] for other in kept if other["category_id"] == item["category_id"]]
                if not same_class or not np.any(box_iou_xywh(item["bbox"], same_class) > NMS_IOU):
                    kept.append(item)
                if len(kept) == max_candidates:
                    break
            ordered = kept
        result.extend(ordered[:max_candidates])
    return result


def match_detections(ground_truth, predictions, drone_category_id, bird_category_id=None):
    image_lookup = {image["id"]: image for image in ground_truth["images"]}
    categories = {category["id"] for category in ground_truth["categories"]}
    if drone_category_id not in categories:
        raise ValueError("The requested drone category is absent from ground truth")
    if bird_category_id is not None and bird_category_id not in categories:
        raise ValueError("The requested bird category is absent from ground truth")
    if bird_category_id == drone_category_id:
        raise ValueError("Drone and bird categories must differ")
    targets = {image_id: [] for image_id in image_lookup}
    birds = {image_id: [] for image_id in image_lookup}
    target_count = 0
    few_count = 0
    for index, annotation in enumerate(ground_truth["annotations"]):
        if annotation["category_id"] in (drone_category_id, bird_category_id) and (annotation.get("iscrowd", 0) or annotation.get("ignore", 0)):
            raise ValueError("FROC crowd/ignore matching is not defined by this protocol; use AP-only evaluation or ordinary FROC annotations")
        if annotation["category_id"] == drone_category_id:
            image = image_lookup[annotation["image_id"]]
            side = 640.0 * math.sqrt(annotation["bbox"][2] * annotation["bbox"][3]) / max(image["width"], image["height"])
            targets[annotation["image_id"]].append((index, annotation, side < 16.0))
            target_count += 1
            few_count += side < 16.0
        elif bird_category_id is not None and annotation["category_id"] == bird_category_id:
            birds[annotation["image_id"]].append(annotation["bbox"])
    used = {image_id: set() for image_id in image_lookup}
    detections = sorted(
        (item for item in predictions if item["category_id"] == drone_category_id),
        key=lambda item: (-item["score"], item["_index"]),
    )
    records = []
    for detection in detections:
        image_id = detection["image_id"]
        available = [target for target in targets[image_id] if target[0] not in used[image_id]]
        matched = None
        if available:
            overlaps = box_iou_xywh(detection["bbox"], [target[1]["bbox"] for target in available])
            best = int(np.argmax(overlaps))
            if overlaps[best] >= MATCH_IOU:
                matched = available[best]
                used[image_id].add(matched[0])
        bird_overlap = matched is None and bool(np.any(box_iou_xywh(detection["bbox"], birds[image_id]) >= MATCH_IOU))
        records.append({
            "image_id": image_id,
            "prediction_index": detection["_index"],
            "score": detection["score"],
            "tp": int(matched is not None),
            "few_tp": int(matched is not None and matched[2]),
            "fp": int(matched is None),
            "bird_overlap_fp": int(bird_overlap),
            "matched_annotation_id": matched[1]["id"] if matched is not None else None,
        })
    return {
        "image_count": len(image_lookup),
        "drone_gt_count": target_count,
        "few_pixel_gt_count": few_count,
        "bird_metric_available": bird_category_id is not None,
        "records": records,
    }


def _point(matching, threshold, tp, few_tp, fp, bird_fp):
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "few_tp": int(few_tp),
        "fp": int(fp),
        "bird_overlap_fp": int(bird_fp) if matching["bird_metric_available"] else None,
        "recall": tp / matching["drone_gt_count"] if matching["drone_gt_count"] else None,
        "few_pixel_recall": few_tp / matching["few_pixel_gt_count"] if matching["few_pixel_gt_count"] else None,
        "fppi": fp / matching["image_count"],
        "bird_overlap_fppi": bird_fp / matching["image_count"] if matching["bird_metric_available"] else None,
    }


def evaluate_at_threshold(matching, threshold):
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    totals = [0, 0, 0, 0]
    for record in matching["records"]:
        if record["score"] >= threshold:
            for index, key in enumerate(("tp", "few_tp", "fp", "bird_overlap_fp")):
                totals[index] += record[key]
    return _point(matching, threshold, *totals)


def operating_curve(matching):
    points = [_point(matching, REJECT_ALL_THRESHOLD, 0, 0, 0, 0)]
    records = matching["records"]
    totals = [0, 0, 0, 0]
    for index, record in enumerate(records):
        for position, key in enumerate(("tp", "few_tp", "fp", "bird_overlap_fp")):
            totals[position] += record[key]
        if index == len(records) - 1 or records[index + 1]["score"] != record["score"]:
            points.append(_point(matching, record["score"], *totals))
    return points


def select_threshold(points, budget=0.05):
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("FPPI budget must be finite and nonnegative")
    valid = [point for point in points if point["fppi"] <= budget]
    if not valid:
        raise ValueError("Operating points must include a reject-all threshold")
    return dict(max(valid, key=lambda point: (point["recall"] if point["recall"] is not None else 0.0, -point["fppi"], point["threshold"])))


def sweep_operating_points(matching, budgets=(0.01, 0.05, 0.10), recall_target=0.90):
    if not 0 <= recall_target <= 1:
        raise ValueError("Recall target must lie in [0, 1]")
    points = operating_curve(matching)
    attainable = [point for point in points if point["recall"] is not None and point["recall"] >= recall_target]
    return {
        "operating_points": [{"fppi_budget": float(budget), **select_threshold(points, float(budget))} for budget in budgets],
        "recall_target": float(recall_target),
        "minimum_fppi_at_recall_target": min((point["fppi"] for point in attainable), default=None),
        "curve": points,
    }


def coco_ap(ground_truth, predictions, category_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = validate_ground_truth(ground_truth)
    requested = list(category_ids) if category_ids is not None else [category["id"] for category in ground_truth["categories"]]
    if not requested or not set(requested).issubset({category["id"] for category in ground_truth["categories"]}):
        raise ValueError("AP requires one or more valid category identifiers")
    ignored = {annotation["id"]: bool(annotation.get("ignore", 0)) for annotation in ground_truth["annotations"]}

    class PreserveIgnoreCOCOeval(COCOeval):
        def _prepare(self):
            super()._prepare()
            for annotations in self._gts.values():
                for annotation in annotations:
                    annotation["ignore"] = bool(annotation.get("ignore", 0) or ignored.get(annotation["id"], False))

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = ground_truth
        coco.createIndex()
        clean = [{key: item[key] for key in ("image_id", "category_id", "bbox", "score")} for item in predictions]
        if clean:
            detections = coco.loadRes(clean)
        else:
            detections = COCO()
            detections.dataset = {"images": copy.deepcopy(ground_truth["images"]), "categories": copy.deepcopy(ground_truth["categories"]), "annotations": []}
            detections.createIndex()
        evaluator = PreserveIgnoreCOCOeval(coco, detections, "bbox")
        evaluator.params.imgIds = [image["id"] for image in ground_truth["images"]]
        evaluator.params.catIds = requested
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.params.recThrs = np.linspace(0, 1, 101)
        evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
        evaluator.evaluate()
        evaluator.accumulate()
        precision = evaluator.eval["precision"][:, :, :, 0, 2]

    def average(values):
        valid = values[values >= 0]
        return float(valid.mean() * 100.0) if len(valid) else None

    return {
        "ap": average(precision),
        "ap50": average(precision[0]),
        "ap75": average(precision[5]),
        "scale": "0-100",
        "category_ids": requested,
        "max_dets": 100,
        "area_range": "all",
        "recall_points": 101,
        "crowd_and_ignore_preserved": True,
    }
