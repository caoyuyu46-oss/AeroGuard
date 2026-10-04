import argparse

from common import local_path, set_quality_eta, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", choices=("onnx", "engine", "torchscript"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--quality-eta", type=float)
    args = parser.parse_args()
    setup_runtime()
    from ultralytics import YOLO

    model = YOLO(str(local_path(args.weights)))
    set_quality_eta(model, args.quality_eta)
    options = dict(format=args.format, imgsz=args.imgsz, batch=1, half=args.half,
                   device=args.device, dynamic=False)
    if args.format in {"onnx", "engine"}:
        options.update(opset=17, simplify=False)
    result = model.export(**options)
    print(result)


if __name__ == "__main__":
    main()
