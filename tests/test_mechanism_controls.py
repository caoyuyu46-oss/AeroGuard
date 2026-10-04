import torch

from ultralytics.nn.modules.aeroguard import BDRP, RDET, SJRHLevel, _area_resize


def test_receiver_condition_ablation_preserves_parameter_count():
    torch.manual_seed(17)
    receiver = torch.randn(1, 8, 4, 4)
    shared = (torch.randn(1, 64, 8, 8), torch.randn(1, 64, 8, 8))
    original = RDET(8)
    for mode, channel in (("no_agreement", 1), ("no_salience", 2)):
        altered = RDET(8, conditioning_mode=mode)
        altered.load_state_dict(original.state_dict())
        _, diagnostics = altered.forward_with_diagnostics((shared, receiver))
        assert torch.count_nonzero(diagnostics["conditioning"][:, channel]) == 0
        assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in altered.parameters())


def test_global_bound_is_computed_independently_for_each_image():
    module = BDRP([8, 8, 8], reference_mode="global")
    inputs = [torch.randn(2, 8, 8, 8), torch.randn(2, 8, 2, 2), torch.randn(2, 8, 4, 4)]
    _, diagnostics = module.forward_with_diagnostics(inputs)
    reference = diagnostics["reference_rms"]
    assert reference.shape == (2, 1, 1, 1)
    expected = torch.sqrt(diagnostics["base"].square().mean((1, 2, 3), keepdim=True) + 1e-6)
    torch.testing.assert_close(reference, expected)
    assert torch.all(diagnostics["bounded_delta"].abs() <= 0.35 * reference + 1e-6)


def test_shared_sampling_computes_one_field_with_separate_refiners():
    module = SJRHLevel(8, 2, shared_sampling=True)
    calls = []
    handle = module.cls_sampler.predictor.register_forward_hook(lambda *args: calls.append(1))
    _, diagnostics = module.forward_with_diagnostics(torch.randn(1, 8, 4, 4))
    handle.remove()
    assert len(calls) == 1
    assert module.cls_refine is not module.reg_refine
    assert diagnostics["cls_sampling"] is diagnostics["reg_sampling"]


def test_static_export_area_pooling_equals_native_area_resampling(monkeypatch):
    values = torch.randn(2, 5, 32, 48)
    for size in ((16, 24), (8, 12), (4, 6)):
        expected = torch.nn.functional.interpolate(values, size=size, mode="area")
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        actual = _area_resize(values, size)
        torch.testing.assert_close(actual, expected)
