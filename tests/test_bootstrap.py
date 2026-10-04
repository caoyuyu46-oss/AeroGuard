import json

import pytest

from bootstrap import resample_groups, run_bootstrap, verified_predictions


def fixture():
    ground_truth = {"images": [{"id": 8, "width": 640, "height": 640}, {"id": 9, "width": 640, "height": 640}],
                    "categories": [{"id": 1, "name": "bird"}, {"id": 2, "name": "drone"}],
                    "annotations": [{"id": 50, "image_id": 8, "category_id": 2, "bbox": [0, 0, 8, 8], "area": 64, "iscrowd": 0},
                                    {"id": 51, "image_id": 9, "category_id": 2, "bbox": [0, 0, 8, 8], "area": 64, "iscrowd": 0}]}
    predictions = [{"image_id": index, "category_id": 2, "bbox": [0, 0, 8, 8], "score": 0.9} for index in (8, 9)]
    return ground_truth, predictions


def test_repeated_groups_get_fresh_identifiers_and_full_membership():
    gt, predictions = fixture()
    sampled, outputs = resample_groups(gt, {"model": predictions}, {8: "a", 9: "a"}, ["a", "a"])
    assert [image["id"] for image in sampled["images"]] == [1, 2, 3, 4]
    assert [annotation["id"] for annotation in sampled["annotations"]] == [1, 2, 3, 4]
    assert [item["image_id"] for item in outputs["model"]] == [1, 2, 3, 4]


def test_paired_bootstrap_is_zero_for_identical_models():
    gt, predictions = fixture()
    inputs = {(name, seed): predictions for name in ("reference", "candidate") for seed in (42, 123)}
    thresholds = {key: 0.5 for key in inputs}
    report = run_bootstrap(gt, inputs, thresholds, {8: "a", 9: "b"}, (42, 123), 2, 1, replicates=3)
    assert report["valid_replicates"] == 3
    assert all(metric["point_estimate"] == 0 and metric["ci95"] == [0, 0] for metric in report["metrics"].values())


def test_prediction_identity_cannot_cross_model_or_seed(tmp_path):
    metadata = {"model_id": "a", "seed": 42, "postprocessed": True, "score_floor": 1e-4,
                "max_candidates": 300, "nms_iou": 0.7}
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"predictions": [], "metadata": metadata}))
    assert verified_predictions(path, metadata, 42) == []
    with pytest.raises(ValueError, match="identity or seed"):
        verified_predictions(path, {**metadata, "model_id": "b"}, 42)
    with pytest.raises(ValueError, match="identity or seed"):
        verified_predictions(path, metadata, 123)
