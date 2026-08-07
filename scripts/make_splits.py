"""Build stratified train/val/test splits + a provenance manifest.

Splits are assigned by SHA-256 of the file's relative path, NOT by
os.listdir() order. That makes them stable across machines and immune to
filesystem ordering.

    python scripts/make_splits.py
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SOURCE_URL = "https://github.com/phelber/EuroSAT"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket(rel_path: str) -> str:
    """Deterministic split from the path hash. Same file -> same split, always."""
    digest = hashlib.sha256(rel_path.encode()).hexdigest()
    x = int(digest[:8], 16) / 0xFFFFFFFF
    if x < RATIOS["train"]:
        return "train"
    if x < RATIOS["train"] + RATIOS["val"]:
        return "val"
    return "test"


def main(root: Path) -> None:
    images = root / "images"
    if not images.is_dir():
        raise SystemExit(f"{images} not found. Run scripts/download_data.py first.")

    classes = sorted(p.name for p in images.iterdir() if p.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    splits = defaultdict(list)
    manifest = {}
    per_class = defaultdict(Counter)

    for cls in classes:
        for img in sorted((images / cls).glob("*.jpg")):
            rel = f"{cls}/{img.name}"
            split = bucket(rel)
            splits[split].append({"path": rel, "label": class_to_idx[cls]})
            manifest[rel] = sha256(img)
            per_class[cls][split] += 1

    (root / "splits.json").write_text(json.dumps({
        "classes": classes,
        "class_to_idx": class_to_idx,
        "splits": dict(splits),
    }, indent=2))

    (root / "manifest.json").write_text(json.dumps({
        "dataset": "EuroSAT RGB",
        "version": "v1_eurosat_rgb_64",
        "source": SOURCE_URL,
        "license": "MIT",
        "downloaded": str(date.today()),
        "sha256": manifest,
    }, indent=2))

    print(f"{'class':<24} {'train':>6} {'val':>6} {'test':>6}")
    for cls in classes:
        c = per_class[cls]
        print(f"{cls:<24} {c['train']:>6} {c['val']:>6} {c['test']:>6}")
    print(f"\n{'TOTAL':<24} "
          f"{len(splits['train']):>6} {len(splits['val']):>6} {len(splits['test']):>6}")
    print(f"\nWrote {root/'splits.json'} and {root/'manifest.json'}")
    print("Next: python -m satdiff.train --config configs/v1.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/v1_eurosat_rgb_64"))
    main(ap.parse_args().root)
