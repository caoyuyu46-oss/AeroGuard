import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


def category(row):
    birds, drones = int(row["birds"]), int(row["drones"])
    return "mixed" if birds and drones else "drone" if drones else "bird" if birds else "background"


def traits(row):
    return [
        "source:" + row["source"],
        "scale:" + row["scale_stratum"],
        "brightness:" + row["brightness_bin"],
        "contrast:" + row["contrast_bin"],
        "density:" + row["density_bin"],
        "edge:" + row["edge_target"],
        "negative:" + row["hard_negative_type"],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, target = args.dataset.resolve(), args.output.resolve()
    if source == target or source.is_relative_to(target):
        raise ValueError("The output must not contain the input dataset")
    with (source / "metadata/manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    random.Random(args.seed).shuffle(rows)
    selected, seen_hashes, used_groups, counts = [], set(), set(), Counter()
    sizes = {"train": 150, "val": 25, "test": 25}
    for split, total in sizes.items():
        pool = [r for r in rows if r["split"] == split]
        quota = {"drone": 75, "bird": 50, "background": 25} if split == "train" else {"drone": 13, "bird": 8, "background": 4}
        if any(category(r) == "mixed" for r in pool):
            quota["mixed"] = 1
            quota["drone"] -= 1
        for kind, number in quota.items():
            for _ in range(number):
                eligible = [r for r in pool if category(r) == kind and r["sha256"] not in seen_hashes
                            and (r["source"], r["group_id"]) not in used_groups]
                if not eligible:
                    raise RuntimeError(f"Insufficient independent groups for {split}/{kind}")
                row = max(eligible, key=lambda r: sum(1.0 / (1 + counts[t]) for t in traits(r)))
                selected.append(row)
                seen_hashes.add(row["sha256"])
                used_groups.add((row["source"], row["group_id"]))
                counts.update(traits(row))
        assert sum(r["split"] == split for r in selected) == total
    target.mkdir(parents=True, exist_ok=True)
    manifest, instance_counts, scale_counts, split_counts = [], Counter(), Counter(), Counter()
    for index, row in enumerate(selected, 1):
        split, stem = row["split"], f"ag_{index:06d}"
        original_image, original_label = source / row["image"], source / row["label"]
        image_bytes, label_bytes = original_image.read_bytes(), original_label.read_bytes()
        assert hashlib.sha256(image_bytes).hexdigest() == row["sha256"]
        assert hashlib.sha256(label_bytes).hexdigest() == row["label_sha256"]
        with Image.open(original_image) as image:
            width, height = image.size
            image.verify()
        parsed = []
        for line in label_bytes.decode("utf-8-sig").splitlines():
            values = list(map(float, line.split()))
            if len(values) != 5 or values[0] not in (0.0, 1.0) or not all(0 <= x <= 1 for x in values[1:]):
                raise ValueError(f"Invalid annotation: {row['id']}")
            if min(values[3:]) <= 0:
                raise ValueError(f"Empty annotation: {row['id']}")
            parsed.append(values)
            instance_counts["bird" if values[0] == 0 else "drone"] += 1
            if values[0] == 1:
                e = 640 * (values[3] * width * values[4] * height) ** 0.5 / max(width, height)
                scale_counts["below_8" if e < 8 else "8_to_16" if e < 16 else "16_to_32" if e < 32 else "32_or_larger"] += 1
        image_rel = f"images/{split}/{stem}{original_image.suffix.lower()}"
        label_rel = f"labels/{split}/{stem}.txt"
        for rel, origin in ((image_rel, original_image), (label_rel, original_label)):
            output = target / rel
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, output)
        split_counts[split] += 1
        manifest.append({
            "sample_id": stem, "split": split, "image": image_rel, "label": label_rel,
            "source_collection": row["source"], "original_id": row["id"],
            "source_group": row["group_id"], "class_state": category(row),
            "width": width, "height": height, "birds": int(row["birds"]), "drones": int(row["drones"]),
            "brightness": row["brightness"], "contrast": row["contrast"],
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
        })
    with (target / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = {"images": len(manifest), "splits": dict(split_counts), "instances": dict(instance_counts),
               "class_states": dict(Counter(r["class_state"] for r in manifest)),
               "source_collections": dict(Counter(r["source_collection"] for r in manifest)),
               "drone_size_bins_e640": dict(scale_counts), "independent_source_groups": len(used_groups),
               "selection_seed": args.seed, "original_splits_preserved": True,
               "image_bytes_preserved": True, "annotation_bytes_preserved": True,
               "purpose": "Example dataset for execution checks; not the paper benchmark"}
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
