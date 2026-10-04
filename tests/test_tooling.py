import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark import summarize
from common import ROOT, resolve_data


def test_p95_uses_nearest_rank():
    records = [{"end_to_end_ms": float(i), "engine_ms": float(i) / 2} for i in range(1, 21)]
    result = summarize(records)
    assert result["end_to_end_ms"]["p95"] == 19.0
    assert result["end_to_end_ms"]["mean"] == 10.5
    assert result["end_to_end_fps"] == pytest.approx(1000 / 10.5)


def test_data_paths_resolve_without_global_dataset_settings(tmp_path):
    import yaml

    result = resolve_data(ROOT / "configs/data/aeroguard_sample.yaml", tmp_path)
    data = yaml.safe_load(result.read_text())
    assert Path(data["train"]) == ROOT / "datasets/aeroguard_sample/images/train"
    assert Path(data["val"]).is_dir()
    assert data["names"] == {0: "bird", 1: "drone"}


def test_paired_statistics_and_seed_mismatch(tmp_path):
    source, output = tmp_path / "metrics.csv", tmp_path / "report.json"
    source.write_text("model,seed,metric,value\nbaseline,42,ap,1\nbaseline,123,ap,4\nfull,42,ap,3\nfull,123,ap,6\n")
    command = [sys.executable, str(ROOT / "tools/aggregate.py"), "--input", str(source), "--reference", "baseline", "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report["paired_comparisons"][0]["mean_difference"] == 2.0
    assert report["paired_comparisons"][0]["paired_95_ci"] == [2.0, 2.0]
    source.write_text(source.read_text().replace("full,123", "full,456"))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "identical seeds" in result.stderr


def test_sample_integrity():
    import csv
    import hashlib

    base = ROOT / "datasets/aeroguard_sample"
    with (base / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert len({(row["source_collection"], row["source_group"]) for row in rows}) == 200
    assert len({row["image_sha256"] for row in rows}) == 200
    for row in rows:
        assert hashlib.sha256((base / row["image"]).read_bytes()).hexdigest() == row["image_sha256"]
        assert hashlib.sha256((base / row["label"]).read_bytes()).hexdigest() == row["label_sha256"]
