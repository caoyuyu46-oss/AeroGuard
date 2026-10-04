import json
from types import SimpleNamespace

import pytest
import torch

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator


def validator(categories):
    instance = DetectionValidator.__new__(DetectionValidator)
    instance.args = SimpleNamespace(paper_evaluation=True, checkpoint_class="drone", split="val")
    instance.data = {}
    instance.nc, instance.niou = len(categories), 10
    instance.names = dict(enumerate(categories))
    instance.metrics = SimpleNamespace(results_dict={"fitness": 0.17}, process=lambda **kwargs: None)
    instance.stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
    instance.paper_ground_truth = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 640, "height": 640}],
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(categories)],
        "annotations": [{"id": index + 1, "image_id": 1, "category_id": index + 1,
                         "bbox": [index * 30, 0, 10, 10], "area": 100, "iscrowd": 0} for index in range(len(categories))],
    }
    instance.paper_predictions = []
    return instance


def test_checkpoint_selects_drone_ap_instead_of_category_mean():
    instance = validator(["bird", "drone"])
    instance.paper_predictions = [{"image_id": 1, "category_id": 2, "bbox": [30, 0, 10, 10], "score": 0.9}]
    result = instance.get_stats()
    assert result["fitness"] == pytest.approx(1)
    assert result["metrics/paper_AP50-95(B)"] == pytest.approx(1)


def test_external_checkpoint_uses_mean_ap_over_its_classes():
    instance = validator(["aircraft", "ship"])
    instance.paper_predictions = [{"image_id": 1, "category_id": 2, "bbox": [30, 0, 10, 10], "score": 0.9}]
    assert instance.get_stats()["fitness"] == pytest.approx(0.5)


def test_empty_predictions_have_zero_selection_ap():
    assert validator(["drone"]).get_stats()["fitness"] == 0


def test_tied_fitness_retains_earlier_epoch_even_at_zero():
    trainer = SimpleNamespace(best_fitness=None, epoch=0, loss=torch.tensor(1.0), validator=lambda _: {"fitness": 0.0})
    BaseTrainer.validate(trainer)
    assert trainer._best_epoch == 0
    trainer.epoch = 1
    BaseTrainer.validate(trainer)
    assert trainer._best_epoch == 0
    trainer.validator = lambda _: {"fitness": 1e-8}
    trainer.epoch = 2
    BaseTrainer.validate(trainer)
    assert trainer._best_epoch == 2
    assert trainer.best_fitness == 1e-8


def test_original_coco_ids_and_ignore_information_survive_mapping(tmp_path):
    instance = validator(["drone"])
    original = {"images": [{"id": 91, "file_name": "sample.jpg", "width": 640, "height": 640}],
                "categories": [{"id": 8, "name": "drone"}],
                "annotations": [{"id": 77, "image_id": 91, "category_id": 8, "bbox": [0, 0, 10, 10],
                                 "iscrowd": 1, "ignore": 1}]}
    path = tmp_path / "val.json"
    path.write_text(json.dumps(original))
    instance.data = {"path": str(tmp_path), "annotations": {"val": "val.json"}}
    instance.paper_predictions = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
    ground_truth, predictions = instance._paper_annotations()
    assert ground_truth == original
    assert predictions[0]["image_id"] == 91
    assert predictions[0]["category_id"] == 8


def test_nested_scene_paths_map_unique_suffixes_and_keep_empty_frames(tmp_path):
    instance = validator(["drone"])
    original = {"images": [{"id": 91, "file_name": "sceneA/frame.jpg", "width": 640, "height": 640},
                           {"id": 92, "file_name": "sceneB\\frame.jpg", "width": 640, "height": 640}],
                "categories": [{"id": 8, "name": "drone"}],
                "annotations": [{"id": 77, "image_id": 91, "category_id": 8, "bbox": [0, 0, 10, 10]}]}
    (tmp_path / "val.json").write_text(json.dumps(original))
    instance.data = {"path": str(tmp_path), "val": "images/val", "annotations": {"val": "val.json"}}
    instance.paper_ground_truth["images"] = [{"id": index, "file_name": str(tmp_path / "images/val" / scene / "frame.jpg"),
                                              "width": 640, "height": 640} for index, scene in ((1, "sceneA"), (2, "sceneB"))]
    instance.paper_predictions = [{"image_id": index, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9} for index in (1, 2)]
    ground_truth, predictions = instance._paper_annotations()
    assert ground_truth == original
    assert [item["image_id"] for item in predictions] == [91, 92]
    assert len(ground_truth["images"]) == 2
    assert len(ground_truth["annotations"]) == 1


@pytest.mark.parametrize("actual_path", ["frame.jpg", "not_sceneA/frame.jpg"])
def test_ambiguous_basenames_and_partial_component_suffixes_are_rejected(tmp_path, actual_path):
    instance = validator(["drone"])
    original = {"images": [{"id": 91, "file_name": "sceneA/frame.jpg", "width": 640, "height": 640},
                           {"id": 92, "file_name": "sceneB/frame.jpg", "width": 640, "height": 640}],
                "categories": [{"id": 8, "name": "drone"}], "annotations": []}
    (tmp_path / "val.json").write_text(json.dumps(original))
    instance.data = {"path": str(tmp_path), "annotations": {"val": "val.json"}}
    instance.paper_ground_truth["images"][0]["file_name"] = str(tmp_path / "images/val" / actual_path)
    with pytest.raises(ValueError, match="matched uniquely"):
        instance._paper_annotations()


@pytest.mark.parametrize("model_name,annotation_name", [("drone", "UAV"), ("uav", "Drone")])
def test_drone_uav_aliases_map_original_category_ids(tmp_path, model_name, annotation_name):
    instance = validator([model_name])
    original = {"images": [{"id": 91, "file_name": "sample.jpg", "width": 640, "height": 640}],
                "categories": [{"id": 8, "name": annotation_name}],
                "annotations": [{"id": 77, "image_id": 91, "category_id": 8, "bbox": [0, 0, 10, 10]}]}
    (tmp_path / "val.json").write_text(json.dumps(original))
    instance.data = {"path": str(tmp_path), "annotations": {"val": "val.json"}}
    instance.paper_predictions = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
    assert instance._paper_annotations()[1][0]["category_id"] == 8
    assert instance.get_stats()["fitness"] == pytest.approx(1)


def test_ambiguous_drone_uav_categories_are_rejected(tmp_path):
    instance = validator(["drone"])
    original = {"images": [{"id": 91, "file_name": "sample.jpg", "width": 640, "height": 640}],
                "categories": [{"id": 8, "name": "drone"}, {"id": 9, "name": "uav"}], "annotations": []}
    (tmp_path / "val.json").write_text(json.dumps(original))
    instance.data = {"path": str(tmp_path), "annotations": {"val": "val.json"}}
    with pytest.raises(ValueError, match="ambiguously match"):
        instance._paper_annotations()
