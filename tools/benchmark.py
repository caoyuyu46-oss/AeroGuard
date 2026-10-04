import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from common import local_path, resolve_data, setup_runtime


def summarize(records):
    result = {}
    for key in ("end_to_end_ms", "engine_ms"):
        values = np.array([r[key] for r in records], dtype=float)
        result[key] = {"mean": float(values.mean()),
                       "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                       "p95": float(np.sort(values)[math.ceil(0.95 * len(values)) - 1])}
    result["end_to_end_fps"] = 1000 / result["end_to_end_ms"]["mean"]
    return result


def image_paths(data, split):
    entries = data[split] if isinstance(data[split], list) else [data[split]]
    paths = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            paths.extend(p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
        else:
            paths.extend((path.parent / line).resolve() if not Path(line).is_absolute() else Path(line)
                         for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return sorted(set(paths), key=lambda p: p.as_posix())


def worker(args):
    setup_runtime()
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils.ops import non_max_suppression, scale_boxes
    from ultralytics.utils.torch_utils import select_device

    output = local_path(args.output)
    resolved = resolve_data(args.data, output / "data_configs")
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    paths = image_paths(data, args.split)
    if not paths:
        raise ValueError("The selected split has no images")
    device = select_device(args.device)
    backend = AutoBackend(str(local_path(args.weights)), device=device, fp16=args.half).eval()
    transform = LetterBox((args.imgsz, args.imgsz), auto=False, stride=int(backend.stride))
    gpu = device.type == "cuda"
    if gpu:
        torch.cuda.set_stream(torch.cuda.default_stream(device))
    start_event = torch.cuda.Event(enable_timing=True) if gpu else None
    end_event = torch.cuda.Event(enable_timing=True) if gpu else None

    def synchronize():
        if gpu:
            torch.cuda.synchronize(device)

    def measure(image):
        synchronize()
        started = time.perf_counter()
        tensor = torch.from_numpy(np.ascontiguousarray(transform(image=image).transpose(2, 0, 1))).to(device)
        tensor = tensor.half() if backend.fp16 else tensor.float()
        tensor = tensor.unsqueeze(0) / 255.0
        if gpu:
            start_event.record()
        engine_started = time.perf_counter()
        prediction = backend(tensor)
        if gpu:
            end_event.record()
        cpu_engine_ms = (time.perf_counter() - engine_started) * 1000
        boxes = non_max_suppression(prediction, conf_thres=1e-4, iou_thres=0.70, max_det=300, agnostic=False)
        for result in boxes:
            result[:, :4] = scale_boxes(tensor.shape[2:], result[:, :4], image.shape)
        _ = [box.cpu().numpy() for box in boxes]
        synchronize()
        return {"end_to_end_ms": (time.perf_counter() - started) * 1000,
                "engine_ms": start_event.elapsed_time(end_event) if gpu else cpu_engine_ms}

    records = []
    with torch.inference_mode():
        warmup_image = np.asarray(Image.open(paths[0]).convert("RGB")).copy()
        started, count = time.perf_counter(), 0
        while count < args.warmup or time.perf_counter() - started < args.min_warmup_seconds:
            measure(warmup_image)
            count += 1
        warmup_seconds = time.perf_counter() - started
        for image_id, path in enumerate(paths, 1):
            with Image.open(path) as handle:
                image = np.asarray(handle.convert("RGB")).copy()
            record = measure(image)
            if not all(math.isfinite(v) and v > 0 for v in record.values()):
                raise RuntimeError("Invalid timing record")
            records.append({"image_id": image_id, "file_name": path.name, **record})
    report = {"process": args.worker, "images": len(paths), "warmup_iterations": count,
              "warmup_seconds": warmup_seconds, "device": str(device),
              "device_name": torch.cuda.get_device_name(device) if gpu else "CPU",
              "engine_clock": "CUDA events on the default stream" if gpu else "Monotonic wall clock",
              "precision": "FP16" if backend.fp16 else "FP32", "batch_size": 1, "imgsz": args.imgsz,
              "cuda_graphs": False, "summary": summarize(records), "records": records}
    (output / f"process_{args.worker}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--min-warmup-seconds", type=float, default=5.0)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--output", default="runs/benchmark")
    args = parser.parse_args()
    if args.warmup < 200 or args.min_warmup_seconds < 5 or args.processes < 1:
        raise ValueError("Use at least 200 warmup iterations, five warmup seconds, and one process")
    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    for process in range(1, args.processes + 1):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(process),
                   "--weights", str(local_path(args.weights)), "--data", str(local_path(args.data)),
                   "--split", args.split, "--device", args.device, "--imgsz", str(args.imgsz),
                   "--warmup", str(args.warmup), "--min-warmup-seconds", str(args.min_warmup_seconds),
                   "--output", str(output)]
        if args.half:
            command.append("--half")
        subprocess.run(command, check=True)
    reports = [json.loads((output / f"process_{i}.json").read_text(encoding="utf-8")) for i in range(1, args.processes + 1)]
    records = [record for report in reports for record in report["records"]]
    summary = {"processes": args.processes, "pooled_frames": len(records), "summary": summarize(records),
               "scope": "RGB image in memory through CPU-readable boxes; file reading and RGB decoding excluded",
               "ordering": "Stable filename order mapped to ascending image_id",
               "precision": reports[0]["precision"], "device": reports[0]["device_name"],
               "engine_clock": reports[0]["engine_clock"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
