from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.aeroguard_loss import AeroGuardDetectionLoss


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = sorted((ROOT / "configs" / "models").glob("*.yaml"))
torch.set_num_threads(2)


def build_model(name="aeroguard"):
    torch.manual_seed(42)
    model = DetectionModel(str(ROOT / "configs" / "models" / f"{name}.yaml"), verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    return model


def make_batch(positive):
    batch = {"img": torch.rand(2, 3, 128, 128)}
    if positive:
        batch.update(
            batch_idx=torch.tensor([0.0, 1.0]),
            cls=torch.tensor([[1.0], [0.0]]),
            bboxes=torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.4, 0.4, 0.3, 0.3]]),
        )
    else:
        batch.update(batch_idx=torch.empty(0), cls=torch.empty(0, 1), bboxes=torch.empty(0, 4))
    return batch


@pytest.mark.parametrize("config", MODEL_CONFIGS, ids=lambda path: path.stem)
def test_every_configuration_builds_with_finite_detections(config):
    model = build_model(config.stem).eval()
    expected_strides = [4, 8, 16, 32] if config.stem == "p2_control" else [8, 16, 32]
    assert model.stride.tolist() == expected_strides
    with torch.inference_mode():
        output, raw = model(torch.rand(1, 3, 128, 128))
    expected_locations = sum((128 // stride) ** 2 for stride in expected_strides)
    assert output.shape == (1, 6, expected_locations)
    assert len(raw) == len(expected_strides)
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(level).all() for level in raw)


@pytest.mark.parametrize("positive", [True, False], ids=["foreground", "empty_ground_truth"])
def test_full_model_loss_backward_and_empty_frame_semantics(positive):
    model = build_model().train()
    criterion = AeroGuardDetectionLoss(model)
    batch = make_batch(positive)
    total, components = criterion(model(batch["img"]), batch)
    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    assert components.shape == (4,)
    assert torch.isfinite(components).all()
    if positive:
        assert criterion.last_diagnostics["positive_count"] > 0
        assert components[0] > 0
        assert components[2] > 0
    else:
        assert criterion.last_diagnostics["positive_count"] == 0
        assert criterion.last_diagnostics["negative_count"] == 2 * (16**2 + 8**2 + 4**2)
        assert components[0] == 0
        assert components[2] == 0
        assert criterion.last_diagnostics["quality_positive"] == 0
        assert criterion.last_diagnostics["quality_negative"] > 0
    assert components[3] > 0
    total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    quality_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "quality_pred" in name and parameter.grad is not None
    ]
    assert quality_gradients and any(gradient.abs().sum() > 0 for gradient in quality_gradients)


@pytest.mark.parametrize("positive", [True, False], ids=["foreground", "empty_ground_truth"])
def test_quality_changes_total_by_effective_gain_3_75(positive):
    model = build_model().train()
    batch = make_batch(positive)
    raw = model(batch["img"])
    criterion = AeroGuardDetectionLoss(model)
    supervised_total, supervised_components = criterion(raw, batch)
    diagnostics = dict(criterion.last_diagnostics)
    unscaled_quality = diagnostics["quality_positive"] + 0.05 * diagnostics["quality_negative"]
    criterion.quality_supervision = False
    detection_total, detection_components = criterion(raw, batch)
    torch.testing.assert_close(supervised_components[:3], detection_components[:3], rtol=0, atol=0)
    assert detection_components[3] == 0
    assert supervised_components[3].item() == pytest.approx(3.75 * unscaled_quality, rel=1e-6, abs=1e-7)
    assert (supervised_total - detection_total).item() == pytest.approx(
        2 * 3.75 * unscaled_quality, rel=2e-4, abs=2e-4
    )


def test_quality_ranking_toggle_preserves_same_checkpoint_boxes():
    model = build_model().eval()
    head = model.model[-1]
    with torch.inference_mode():
        ranked, raw = model(torch.rand(1, 3, 128, 128))
        head.set_quality_scoring(True, 0.0)
        unranked = head._inference(raw)
    quality = torch.cat([level[:, -1:].flatten(2) for level in raw], dim=2).sigmoid()
    torch.testing.assert_close(ranked[:, :4], unranked[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(ranked[:, 4:], unranked[:, 4:] * quality.sqrt())
    assert torch.all(ranked[:, 4:] <= unranked[:, 4:])
