import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics.nn.tasks import DetectionModel


def test_fixed_onnx_export_matches_native_predictions(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.set_num_threads(2)
    torch.manual_seed(91)
    root = Path(__file__).resolve().parents[1]
    model = DetectionModel(str(root / "configs/models/aeroguard.yaml"), nc=2, verbose=False).eval().fuse(verbose=False)
    model.model[-1].export = True
    value = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        expected = model(value).numpy()
    target = tmp_path / "model.onnx"
    options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with torch.no_grad():
        torch.onnx.export(model, value, str(target), input_names=["images"], output_names=["output0"],
                          opset_version=17, do_constant_folding=True, **options)
    onnx.checker.check_model(onnx.load(target))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(target), sess_options=session_options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": value.numpy()})[0]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
