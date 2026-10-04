import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(dict)
    with args.input.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key, seed, value = (row["model"], row["metric"]), int(row["seed"]), float(row["value"])
            if seed in groups[key] or not np.isfinite(value):
                raise ValueError("Each model/metric/seed must have exactly one finite value")
            groups[key][seed] = value
    summaries, comparisons = [], []
    for (model, metric), by_seed in sorted(groups.items()):
        values = np.array(list(by_seed.values()))
        summaries.append({"model": model, "metric": metric, "seeds": sorted(by_seed),
                          "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None})
        if args.reference and model != args.reference and (args.reference, metric) in groups:
            reference = groups[(args.reference, metric)]
            if set(reference) != set(by_seed):
                raise ValueError(f"Matched-seed comparison requires identical seeds: {model}/{metric}")
            differences = np.array([by_seed[s] - reference[s] for s in sorted(by_seed)])
            interval = None
            if len(differences) > 1:
                half = t.ppf(0.975, len(differences) - 1) * differences.std(ddof=1) / len(differences) ** 0.5
                interval = [float(differences.mean() - half), float(differences.mean() + half)]
            comparisons.append({"model": model, "reference": args.reference, "metric": metric,
                                "seeds": sorted(by_seed), "mean_difference": float(differences.mean()), "paired_95_ci": interval})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summaries": summaries, "paired_comparisons": comparisons}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
