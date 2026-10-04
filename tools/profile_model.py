import argparse
import copy
import json

import torch

from common import local_path, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="runs/profile.json")
    args = parser.parse_args()
    setup_runtime()
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops

    model = DetectionModel(str(local_path(args.model)), ch=3, nc=args.classes).eval()
    torch.set_num_threads(2)
    with torch.no_grad():
        output = model(torch.zeros(1, 3, args.imgsz, args.imgsz))
    decoded = output[0] if isinstance(output, tuple) else output
    fused = copy.deepcopy(model).fuse(verbose=False)
    unfused_flops, fused_flops = get_flops(model, args.imgsz), get_flops(fused, args.imgsz)
    report = {"classes": args.classes, "input_size": args.imgsz,
              "parameters": sum(p.numel() for p in model.parameters()),
              "thop_gflops_unfused": unfused_flops if unfused_flops > 0 else None,
              "thop_gflops_fused": fused_flops if fused_flops > 0 else None,
              "thop_available": unfused_flops > 0 and fused_flops > 0,
              "decoded_shape": list(decoded.shape), "strides": model.stride.tolist(),
              "finite_output": bool(torch.isfinite(decoded).all()),
              "counting": "THOP registered operations; multiply-add counts as two operations",
              "functional_operations": "Functional resampling and elementwise operations may not be counted by THOP"}
    destination = local_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
