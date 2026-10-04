import copy
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate import build_parser, data_ground_truth, run
from tools.common import set_quality_eta
from tools.metrics import (
    REJECT_ALL_THRESHOLD,
    coco_ap,
    evaluate_at_threshold,
    match_detections,
    operating_curve,
    prepare_predictions,
    select_threshold,
    sweep_operating_points,
    validate_ground_truth,
)


def dataset(image_count=20, annotations=None):
    return validate_ground_truth({
        "images": [{"id": index + 1, "width": 640, "height": 640, "file_name": f"{index + 1}.png"} for index in range(image_count)],
        "annotations": annotations if annotations is not None else [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 8, 8]}],
        "categories": [{"id": 1, "name": "bird"}, {"id": 2, "name": "drone"}],
    })


def prediction(image_id=1, category=2, box=None, score=0.8):
    return {"image_id": image_id, "category_id": category, "bbox": box or [10, 10, 8, 8], "score": score}


def match(data, predictions):
    return match_detections(data, prepare_predictions(predictions, data), 2, 1)


class PaperEvaluationTests(unittest.TestCase):
    def test_background_images_remain_in_fppi_denominator(self):
        data = dataset()
        matching = match(data, [prediction(), prediction(2, box=[50, 50, 8, 8], score=0.7)])
        point = evaluate_at_threshold(matching, 0.7)
        self.assertEqual(point["recall"], 1.0)
        self.assertEqual(point["fppi"], 0.05)
        self.assertEqual(matching["image_count"], 20)

    def test_cross_class_cap_precedes_drone_extraction(self):
        data = dataset(1)
        predictions = [prediction(category=1, score=0.9) for _ in range(300)] + [prediction(score=0.8)]
        retained = prepare_predictions(predictions, data)
        self.assertEqual(len(retained), 300)
        self.assertTrue(all(item["category_id"] == 1 for item in retained))
        self.assertEqual(evaluate_at_threshold(match_detections(data, retained, 2, 1), 0)["recall"], 0)

    def test_raw_nms_is_class_aware_and_optional(self):
        data = dataset(1)
        predictions = [prediction(score=0.9), prediction(score=0.8), prediction(category=1, score=0.7)]
        self.assertEqual(len(prepare_predictions(predictions, data)), 3)
        raw = prepare_predictions(predictions, data, raw_predictions=True)
        self.assertEqual([item["category_id"] for item in raw], [2, 1])

    def test_score_ties_form_indivisible_threshold_groups(self):
        data = dataset(1)
        matching = match(data, [prediction(score=0.8), prediction(box=[50, 50, 8, 8], score=0.8)])
        curve = operating_curve(matching)
        self.assertEqual(len(curve), 2)
        selected = select_threshold(curve, 0.0)
        self.assertEqual(selected["threshold"], REJECT_ALL_THRESHOLD)
        self.assertEqual(selected["recall"], 0)

    def test_stable_prediction_and_gt_tie_breaks(self):
        annotations = [
            {"id": 9, "image_id": 1, "category_id": 2, "bbox": [10, 10, 8, 8]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [10, 10, 8, 8]},
        ]
        records = match(dataset(1, annotations), [prediction(), prediction(), prediction()])["records"]
        self.assertEqual([row["prediction_index"] for row in records], [0, 1, 2])
        self.assertEqual([row["matched_annotation_id"] for row in records], [9, 2, None])
        self.assertEqual([row["tp"] for row in records], [1, 1, 0])

    def test_bird_overlap_is_a_subset_of_false_positives(self):
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 8, 8]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 8, 8]},
        ]
        matching = match(dataset(2, annotations), [prediction(score=0.9), prediction(score=0.8), prediction(2, score=0.7)])
        point = evaluate_at_threshold(matching, 0)
        self.assertEqual(point["tp"], 1)
        self.assertEqual(point["fp"], 2)
        self.assertEqual(point["bird_overlap_fp"], 1)
        self.assertEqual(point["bird_overlap_fppi"], 0.5)

    def test_few_pixel_recall_uses_original_image_scale_and_shared_threshold(self):
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [10, 10, 32, 32]},
        ]
        data = dataset(2, annotations)
        data["images"][0]["width"] = 1280
        matching = match(data, [prediction(box=[10, 10, 20, 20], score=0.7), prediction(2, box=[10, 10, 32, 32], score=0.9)])
        point = evaluate_at_threshold(matching, 0.8)
        self.assertEqual(matching["few_pixel_gt_count"], 1)
        self.assertEqual(point["recall"], 0.5)
        self.assertEqual(point["few_pixel_recall"], 0)

    def test_threshold_selection_ties_and_unattainable_recall(self):
        annotations = [{"id": index + 1, "image_id": index + 1, "category_id": 2, "bbox": [10, 10, 8, 8]} for index in range(2)]
        matching = match(dataset(20, annotations), [prediction(score=0.9), prediction(3, box=[40, 40, 8, 8], score=0.7)])
        sweep = sweep_operating_points(matching)
        self.assertIsNone(sweep["minimum_fppi_at_recall_target"])
        selected = select_threshold(sweep["curve"], 0.05)
        self.assertEqual(selected["threshold"], 0.9)
        self.assertEqual(selected["fppi"], 0)
        duplicate = [dict(selected, threshold=0.8), selected]
        self.assertEqual(select_threshold(duplicate)["threshold"], 0.9)

    def test_empty_predictions_and_no_target_recall(self):
        matching = match(dataset(), [])
        self.assertEqual(evaluate_at_threshold(matching, 0)["recall"], 0)
        self.assertEqual(select_threshold(operating_curve(matching))["threshold"], REJECT_ALL_THRESHOLD)
        no_targets = match(dataset(2, []), [])
        self.assertIsNone(evaluate_at_threshold(no_targets, 0)["recall"])
        self.assertIsNone(coco_ap(dataset(2, []), [], [2])["ap"])
        self.assertEqual(coco_ap(dataset(), [], [2])["ap"], 0)

    def test_coco_ap_enforces_max_dets_100(self):
        data = dataset(1)
        predictions = [prediction(box=[100 + index * 2, 100, 1, 1], score=0.9) for index in range(100)]
        predictions.append(prediction(score=0.8))
        prepared = prepare_predictions(predictions, data)
        self.assertEqual(len(prepared), 101)
        ap = coco_ap(data, prepared, [2])
        self.assertEqual(ap["max_dets"], 100)
        self.assertEqual(ap["ap"], 0)

    def test_ap_only_multiclass_and_original_ignore_flags(self):
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 8, 8]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [30, 30, 8, 8], "ignore": 1},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [50, 50, 8, 8]},
        ]
        data = dataset(1, annotations)
        original = copy.deepcopy(data)
        prepared = prepare_predictions([prediction()], data)
        self.assertAlmostEqual(coco_ap(data, prepared, [2])["ap"], 100)
        self.assertAlmostEqual(coco_ap(data, prepared)["ap"], 50)
        self.assertEqual(data, original)
        with self.assertRaisesRegex(ValueError, "crowd/ignore"):
            match_detections(data, prepared, 2, 1)

    def test_invalid_prediction_references_and_nonfinite_scores(self):
        data = dataset(1)
        for predictions in ([prediction(9)], [prediction(category=3)], [prediction(score=math.nan)]):
            with self.assertRaises(ValueError):
                prepare_predictions(predictions, data)

    def test_zero_area_predictions_remain_false_positives(self):
        data = dataset(1)
        prepared = prepare_predictions([prediction(box=[10, 10, 0, 8], score=0.9), prediction(box=[10, 10, 8, 0], score=0.8), prediction(score=0.7)], data)
        self.assertEqual(len(prepared), 3)
        point = evaluate_at_threshold(match_detections(data, prepared, 2, 1), 0)
        self.assertEqual((point["tp"], point["fp"], point["bird_overlap_fp"]), (1, 2, 0))
        self.assertEqual(point["fppi"], 2)
        self.assertTrue(math.isfinite(coco_ap(data, prepared, [2])["ap"]))
        with self.assertRaisesRegex(ValueError, "nonnegative size"):
            prepare_predictions([prediction(box=[10, 10, -1, 8])], data)

    def test_quality_eta_requires_weights_and_native_head(self):
        for value in ("0", "-1", "nan", "inf"):
            args = build_parser().parse_args(["--predictions", "unused.json", "--seed", "42", "--output", "unused", "--quality-eta", value])
            with self.assertRaisesRegex(ValueError, "requires weights"):
                run(args)
        for value in ("-1", "nan", "inf"):
            args = build_parser().parse_args(["--weights", "unused.pt", "--seed", "42", "--output", "unused", "--quality-eta", value])
            with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                run(args)
        calls = []
        model = SimpleNamespace(model=SimpleNamespace(model=[SimpleNamespace(set_quality_scoring=lambda **kwargs: calls.append(kwargs))]))
        set_quality_eta(model, 0)
        self.assertEqual(calls, [{"enabled": True, "eta": 0}])
        with self.assertRaisesRegex(ValueError, "native SJRH"):
            set_quality_eta(SimpleNamespace(model=SimpleNamespace(model=[object()])), 0.5)

    def test_quality_eta_metadata_identity_and_frozen_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gt = root / "gt.json"
            gt.write_text(json.dumps(dataset()))
            args = build_parser().parse_args(["--annotations", str(gt), "--weights", "local.pt", "--seed", "42", "--split", "val", "--quality-eta", "0", "--output", str(root / "val")])
            with patch("tools.evaluate.weight_predictions", return_value=([prediction()], {"weights_sha256": "abc123", "postprocessed": True})):
                report = run(args)
            self.assertEqual(report["model_id"], "abc123|quality_eta=0")
            metadata = json.loads((root / "val/prediction_metadata.json").read_text())
            self.assertEqual(metadata["quality_eta"], 0)
            self.assertEqual(metadata["seed"], 42)
            self.assertTrue(metadata["postprocessed"])
            self.assertEqual(metadata["max_candidates"], 300)
            self.assertIsInstance(json.loads((root / "val/predictions.json").read_text()), list)
            args.split = "test"
            args.output = str(root / "test")
            args.threshold_from = str(root / "val/threshold.json")
            with patch("tools.evaluate.weight_predictions", return_value=([prediction()], {"weights_sha256": "abc123", "postprocessed": True})):
                self.assertEqual(run(args)["frozen_validation_threshold"]["recall"], 1)
            args.quality_eta = 0.5
            with patch("tools.evaluate.weight_predictions", return_value=([prediction()], {"weights_sha256": "abc123", "postprocessed": True})):
                with self.assertRaisesRegex(ValueError, "model identifier"):
                    run(args)
            args.model_id = "abc123|quality_eta=0"
            with patch("tools.evaluate.weight_predictions", return_value=([prediction()], {"weights_sha256": "abc123", "postprocessed": True})):
                with self.assertRaisesRegex(ValueError, "quality exponent"):
                    run(args)

    def test_offline_sidecar_preserves_native_postprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "gt.json").write_text(json.dumps(dataset()))
            (root / "predictions.json").write_text(json.dumps([prediction()]))
            (root / "prediction_metadata.json").write_text(json.dumps({"seed": 42, "model_id": "native-detector", "postprocessed": True, "native_postprocessing": "nms_free", "class_aware_nms": False}))
            args = build_parser().parse_args(["--annotations", str(root / "gt.json"), "--predictions", str(root / "predictions.json"), "--seed", "42", "--split", "val", "--output", str(root / "val")])
            report = run(args)
            self.assertEqual(report["model_id"], "native-detector")
            metadata = json.loads((root / "val/prediction_metadata.json").read_text())
            self.assertEqual(metadata["native_postprocessing"], "nms_free")
            self.assertFalse(metadata["class_aware_nms"])

    def test_cli_frozen_threshold_does_not_retarget_test_fppi(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            val_gt, val_pred, test_gt, test_pred = (root / name for name in ("val_gt.json", "val_pred.json", "test_gt.json", "test_pred.json"))
            for path, content in ((val_gt, dataset(20)), (val_pred, [prediction(score=0.8)]), (test_gt, dataset(1)), (test_pred, [prediction(score=0.8), prediction(box=[50, 50, 8, 8], score=0.9)])):
                path.write_text(json.dumps(content), encoding="utf-8")
            val = build_parser().parse_args(["--annotations", str(val_gt), "--predictions", str(val_pred), "--seed", "42", "--model-id", "fixture", "--split", "val", "--output", str(root / "val")])
            run(val)
            threshold_file = root / "val/threshold.json"
            self.assertEqual(json.loads(threshold_file.read_text())["threshold"], 0.8)
            test = build_parser().parse_args(["--annotations", str(test_gt), "--predictions", str(test_pred), "--seed", "42", "--model-id", "fixture", "--split", "test", "--threshold-from", str(threshold_file), "--output", str(root / "test")])
            report = run(test)
            self.assertEqual(report["frozen_validation_threshold"]["fppi"], 1)
            self.assertEqual(report["frozen_validation_threshold"]["recall"], 1)
            point = next(point for point in report["score_sweep"]["operating_points"] if point["fppi_budget"] == 0.05)
            self.assertEqual(point["recall"], 0)
            test.seed = 123
            with self.assertRaisesRegex(ValueError, "same seed"):
                run(test)
            test.seed = 42
            test.model_id = "other-checkpoint"
            with self.assertRaisesRegex(ValueError, "model identifier"):
                run(test)
            test.model_id = "fixture"
            threshold = json.loads(threshold_file.read_text())
            threshold["drone_category_id"] = 1
            threshold_file.write_text(json.dumps(threshold))
            with self.assertRaisesRegex(ValueError, "category identifiers"):
                run(test)

    def test_offline_model_identity_and_automatic_weight_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gt = root / "gt.json"
            pred = root / "pred.json"
            gt.write_text(json.dumps(dataset()))
            pred.write_text(json.dumps([prediction()]))
            args = build_parser().parse_args(["--annotations", str(gt), "--predictions", str(pred), "--seed", "42", "--split", "val", "--output", str(root / "offline")])
            with self.assertRaisesRegex(ValueError, "require --model-id"):
                run(args)
            pred.write_text(json.dumps({"predictions": [prediction()], "metadata": {"model_id": "checkpoint-42"}}))
            report = run(args)
            self.assertEqual(report["model_id"], "checkpoint-42")
            weight_args = build_parser().parse_args(["--annotations", str(gt), "--weights", "local.pt", "--seed", "42", "--split", "val", "--output", str(root / "weights")])
            with patch("tools.evaluate.weight_predictions", return_value=([prediction()], {"weights_sha256": "abc123", "postprocessed": True})):
                report = run(weight_args)
            self.assertEqual(report["model_id"], "abc123")
            self.assertEqual(report["validation_selected"]["model_id"], "abc123")

    def test_yolo_loading_preserves_empty_label_backgrounds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images/val").mkdir(parents=True)
            (root / "labels/val").mkdir(parents=True)
            for name in ("a", "b"):
                Image.new("RGB", (128, 64)).save(root / f"images/val/{name}.png")
            (root / "labels/val/a.txt").write_text("1 0.5 0.5 0.125 0.25\n")
            (root / "labels/val/b.txt").write_text("")
            config = root / "data.yaml"
            config.write_text(f"path: {root.as_posix()}\nval: images/val\nnames: [bird, drone]\n")
            gt, metadata = data_ground_truth(config, "val")
            self.assertEqual(len(gt["images"]), 2)
            self.assertEqual(len(gt["annotations"]), 1)
            self.assertEqual(gt["annotations"][0]["category_id"], 2)
            self.assertEqual(gt["annotations"][0]["bbox"], [56, 24, 16, 16])
            self.assertFalse(metadata["crowd_ignore_from_source"])
            (root / "labels/val/b.txt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing label"):
                data_ground_truth(config, "val")

    def test_data_annotations_override_yolo_label_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            annotations = dataset(2)
            annotations["annotations"][0]["iscrowd"] = 1
            (root / "gt.json").write_text(json.dumps(annotations))
            config = root / "data.yaml"
            config.write_text(f"path: {root.as_posix()}\nval: does_not_exist\nannotations:\n  val: gt.json\nnames: [bird, drone]\n")
            gt, metadata = data_ground_truth(config, "val")
            self.assertEqual(gt["annotations"][0]["iscrowd"], 1)
            self.assertEqual(metadata["format"], "COCO")


if __name__ == "__main__":
    unittest.main()
