# AeroGuard

Code for **Controlled Detail Transfer and Grid-Relative Geometry Learning for Visible-Light Anti-UAV Detection**.

This distribution integrates the manuscript's RDET, BDRP, SJRH, and SAGE definitions into the official Ultralytics v8.3.0 source. The upstream commit is `6e43d1e1e5db72afbf686dee6745669bcb124b0a`. The package version is `8.3.0+aeroguard.20261003`.

## Installation

Run commands from the repository root in a fresh Python environment. The manuscript environment uses Python 3.10, PyTorch 2.4.1, torchvision 0.19.1, and CUDA 12.4.

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux or macOS, or `.venv\Scripts\Activate.ps1` in PowerShell. Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e . -r requirements/paper.txt
```

For CPU execution, use the PyTorch CPU wheel index instead of `cu124`. Jetson deployment requires its platform-compatible PyTorch and TensorRT packages.

## Included sample

`datasets/aeroguard_sample` contains 200 images and their labels: 150 training, 25 validation, and 25 test images. Classes are `bird=0` and `drone=1`. The selection preserves each image's original split and includes drone, bird, mixed, and background scenes from six source collections. The sample manifest records provenance and image/label SHA-256 values.

The sample supports installation and execution checks. Full benchmark results require the complete datasets and their frozen splits. Dataset images retain their source-specific rights; see `datasets/aeroguard_sample/DATA_LICENSE.txt`.

## Training

Run a short execution check on the included sample:

```bash
python tools/train.py --config configs/experiments/aeroguard.yaml --data configs/data/aeroguard_sample.yaml --epochs 1 --batch 2 --imgsz 128 --device cpu --workers 0 --no-amp --no-plots --name sample_check
```

Train the full model with the paper recipe:

```bash
python tools/train.py --config configs/experiments/aeroguard.yaml --data configs/data/aeroguard_uavb.yaml --weights yolo11n.pt --seed 42 --device 0 --name aeroguard_seed42
```

The optional `--weights yolo11n.pt` argument downloads the official initialization checkpoint if it is absent. Only matching parameter names and shapes are loaded. Omit `--weights` for random initialization. Use seeds `42, 123, 456, 789, 2024` for five-run comparisons and the first three for three-run comparisons. Each run uses a separate output name.

Model definitions are in `configs/models`, and training recipes are in `configs/experiments`. Data configurations are in `configs/data`. External datasets use their own class maps. Their images should follow `images/{train,val,test}` and `labels/{train,val,test}`, with zero-based YOLO class IDs matching the selected YAML. Original COCO annotations are required when AP evaluation must preserve crowd or ignore regions.

`configs/experiments/manifest.json` lists the available structural and mechanism variants and the explicit definitions of the P2 and capacity controls. It also identifies comparison implementations and historical artifacts that are not bundled.

## Evaluation

Select the operating threshold on validation, then freeze it for testing:

```bash
python tools/evaluate.py --weights runs/train/aeroguard_seed42/weights/best.pt --data configs/data/aeroguard_uavb.yaml --split val --seed 42 --output runs/evaluation/seed42/val
python tools/evaluate.py --weights runs/train/aeroguard_seed42/weights/best.pt --data configs/data/aeroguard_uavb.yaml --split test --seed 42 --threshold-from runs/evaluation/seed42/val/threshold.json --output runs/evaluation/seed42/test
```

Offline evaluation accepts COCO ground truth and prediction JSON files:

```bash
python tools/evaluate.py --annotations annotations/val.json --predictions val_predictions.json --model-id aeroguard_seed42 --split val --seed 42 --output runs/evaluation/offline_val
python tools/evaluate.py --annotations annotations/test.json --predictions test_predictions.json --model-id aeroguard_seed42 --split test --seed 42 --threshold-from runs/evaluation/offline_val/threshold.json --output runs/evaluation/offline_test
```

COCO AP uses `maxDets=100`. Operating-point evaluation retains at most 300 candidates across classes, uses a score floor of `0.0001`, and matches drone predictions at IoU `0.5`. Validation selection maximizes recall under FPPI `0.05`, resolving ties by lower FPPI and then higher threshold. Frozen-threshold test metrics and independent test-score sweeps are reported separately. All evaluation images, including background images, contribute to FPPI.

Offline predictions are assumed to have undergone their detector's native postprocessing. Use `--raw-predictions` only when class-aware NMS is still required. Use `--ap-only` for external multiclass benchmarks. COCO category IDs in generated annotations are the YAML class index plus one.

For the ranking comparison, reuse the same SJRH checkpoint with `--quality-eta 0` and `--quality-eta 0.5`. Select a separate validation threshold for each scoring setting, then retain that threshold on test. The evaluator records checkpoint and scoring identities to prevent mixing threshold files between settings.

## Prediction and export

```bash
python tools/predict.py --weights runs/train/aeroguard_seed42/weights/best.pt --source datasets/aeroguard_sample/images/test --device cpu --output runs/predictions
python tools/export.py --weights runs/train/aeroguard_seed42/weights/best.pt --format onnx --device cpu
python tools/export.py --weights runs/train/aeroguard_seed42/weights/best.pt --format engine --device 0 --half
```

Exports use batch size 1 and fixed input dimensions; the default input size is 640. ONNX and TensorRT export use opset 17. Build TensorRT engines on the target deployment platform.

## Profiling and timing

```bash
python tools/profile_model.py --model configs/models/aeroguard.yaml --classes 2 --imgsz 640 --output runs/profile.json
python tools/benchmark.py --weights runs/train/aeroguard_seed42/weights/best.engine --data configs/data/aeroguard_uavb.yaml --split test --device 0 --half --output runs/benchmark
```

Profiling reports measured parameter counts and THOP operation counts. Timing uses three independent processes, at least 200 warmup iterations and five warmup seconds per process, and every image in the chosen split. Engine and end-to-end timing are separate. The full deployment settings are recorded in `configs/deployment/jetson.yaml`.

## Statistical and feature diagnostics

`tools/aggregate.py` aggregates a CSV with columns `model,seed,metric,value`, reporting means and sample standard deviations. Its `--reference` option computes paired confidence intervals from identical seed sets.

`tools/bootstrap.py` performs the paired source-group bootstrap using five matched seeds and frozen validation thresholds. Supply the original COCO annotations, the exact `image_id,group_id` CSV, and prediction/threshold paths in `seed=path` form. Keep each prediction file beside the `prediction_metadata.json` produced by the evaluator. Its defaults are 312 groups, 2,000 valid resamples, and random seed 20260928. Run `python tools/bootstrap.py --help` for the input arguments. The original 312-group mapping is not included in this distribution.

`tools/feature_ecdf.py` accepts an NPZ containing `increment` (C x H x W), `reference_rms` (H x W), `x_edges`, `y_edges`, `valid_bounds`, `drone_boxes`, and `bird_boxes`. Coordinates use one common spatial system and GT boxes use xyxy order. RDET uses the receiver-projection channel RMS as the reference; BDRP uses its base-fusion RMS. The tool exports regional weighted ECDFs without display interpolation or clipping.

## Validation

```bash
python -m pytest tests -q
```

The release validation record distinguishes local engineering checks from full experimental replication. Trained paper checkpoints, the full benchmark datasets, and external frozen split lists are not bundled. The 200-image sample is not used to substantiate the manuscript's reported accuracy or deployment latency.

For two classes and a 640 input, the model has 2,590,482 parameters and 7.533 GFLOPs under unfused THOP counting, with one multiply-add counted as two operations. The manuscript reports these values as 2.590 M parameters and 7.533 GFLOPs. The capacity control uses standard operators and retains P3-P5 prediction.

## License and citation

Source code is distributed under the upstream AGPL-3.0 license. Upstream legal notices are preserved in `LICENSE`, `NOTICE`, and the corresponding source files. This is a manuscript implementation based on Ultralytics, not an official Ultralytics release. Cite the manuscript using `CITATION.bib`; publication metadata can be added when available.
