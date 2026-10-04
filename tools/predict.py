import argparse

from common import local_path, set_quality_eta, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/predict")
    parser.add_argument("--quality-eta", type=float)
    args = parser.parse_args()
    setup_runtime()
    from ultralytics import YOLO

    model = YOLO(str(local_path(args.weights)))
    set_quality_eta(model, args.quality_eta)
    model.predict(source=str(local_path(args.source)), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                  max_det=300, agnostic_nms=False, device=args.device, save=True, save_txt=True,
                  save_conf=True, project=str(local_path(args.output)), name="detections", exist_ok=True)


if __name__ == "__main__":
    main()
