import argparse
import json

import numpy as np

from common import local_path


def union_area(rectangles):
    boxes = np.asarray(rectangles, dtype=float).reshape(-1, 4)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if not len(boxes):
        return 0.0
    edges = np.unique(boxes[:, [0, 2]])
    area = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        active = boxes[(boxes[:, 0] < right) & (boxes[:, 2] > left)]
        if not len(active):
            continue
        intervals = sorted(active[:, [1, 3]].tolist())
        lower, upper = intervals[0]
        height = 0.0
        for lo, hi in intervals[1:]:
            if lo > upper:
                height += upper - lower
                lower, upper = lo, hi
            else:
                upper = max(upper, hi)
        area += (right - left) * (height + upper - lower)
    return float(area)


def area_weights(x_edges, y_edges, valid_bounds, drone_boxes, bird_boxes):
    x_edges, y_edges = np.asarray(x_edges, float), np.asarray(y_edges, float)
    bounds = np.asarray(valid_bounds, float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)) or np.any(bounds[2:] <= bounds[:2]):
        raise ValueError("Valid bounds must be a nonempty finite xyxy rectangle")
    if len(x_edges) < 2 or len(y_edges) < 2 or not np.all(np.diff(x_edges) > 0) or not np.all(np.diff(y_edges) > 0):
        raise ValueError("Cell edges must be strictly increasing")
    classes = [np.asarray(boxes, float).reshape(-1, 4) for boxes in (drone_boxes, bird_boxes)]
    if not all(np.isfinite(boxes).all() for boxes in classes):
        raise ValueError("Ground-truth coordinates must be finite")
    weights = np.zeros((3, len(y_edges) - 1, len(x_edges) - 1), dtype=float)
    valid = np.zeros(weights.shape[1:], dtype=bool)
    for y, (top, bottom) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        for x, (left, right) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            cell = np.array([max(left, bounds[0]), max(top, bounds[1]), min(right, bounds[2]), min(bottom, bounds[3])])
            if np.any(cell[2:] <= cell[:2]):
                continue
            valid[y, x] = True
            area = np.prod(cell[2:] - cell[:2])
            clipped = []
            for index, boxes in enumerate(classes):
                clip = np.concatenate((np.maximum(boxes[:, :2], cell[:2]), np.minimum(boxes[:, 2:], cell[2:])), axis=1)
                clipped.append(clip)
                weights[index, y, x] = union_area(clip) / area
            weights[2, y, x] = max(0.0, 1.0 - union_area(np.concatenate(clipped, axis=0)) / area)
    return weights, valid


def weighted_ecdf(values, weights):
    values, weights = np.asarray(values, float).ravel(), np.asarray(weights, float).ravel()
    if values.shape != weights.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Values and nonnegative finite weights must have matching shapes")
    keep = weights > 0
    if not keep.any():
        return None
    values, weights = values[keep], weights[keep]
    if not np.isfinite(values).all():
        raise ValueError("Positive-weight values must be finite")
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    unique = values[starts]
    cumulative = np.cumsum(np.add.reduceat(weights, starts)) / weights.sum()
    cumulative[-1] = 1.0
    return {"relative_rms": unique.tolist(), "cdf": cumulative.tolist(), "total_weight": float(weights.sum())}


def spatial_ecdfs(increment, reference_rms, x_edges, y_edges, valid_bounds, drone_boxes, bird_boxes, eps=1e-6):
    increment, reference = np.asarray(increment, float), np.asarray(reference_rms, float)
    if increment.ndim != 3 or reference.shape != increment.shape[1:]:
        raise ValueError("Increment must be CxHxW and reference RMS must be HxW")
    if eps <= 0 or np.any(reference < 0) or not np.isfinite(reference).all():
        raise ValueError("Reference RMS and stabilizer must be valid")
    if increment.shape[1:] != (len(y_edges) - 1, len(x_edges) - 1):
        raise ValueError("Cell edges must describe the native feature grid")
    weights, valid = area_weights(x_edges, y_edges, valid_bounds, drone_boxes, bird_boxes)
    relative = np.sqrt(np.mean(increment**2, axis=0)) / (reference + eps)
    return {name: curve for index, name in enumerate(("Drone", "Bird", "Background"))
            if (curve := weighted_ecdf(relative[valid], weights[index][valid])) is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(local_path(args.input), allow_pickle=False) as data:
        result = spatial_ecdfs(*(data[key] for key in ("increment", "reference_rms", "x_edges", "y_edges",
                                                     "valid_bounds", "drone_boxes", "bird_boxes")))
    target = local_path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
