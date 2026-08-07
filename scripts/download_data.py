"""Download EuroSAT RGB (27,000 images, 64x64, MIT license).

Uses the RGB JPEG release directly. Do NOT use the 13-band .tif version and
convert with rasterio — that step is unnecessary and loses nothing but time.

Deliberately stdlib-only: fetching data should not require a 2 GB deep
learning install.

    python scripts/download_data.py
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

MIRRORS = [
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip",
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
]
EXPECTED_CLASSES = 10
EXPECTED_IMAGES = 27_000


_last_pct = -1


def progress(count, block_size, total):
    """Print only on each 10% step — a \\r bar becomes 200 KB of noise when
    stdout is a pipe rather than a terminal."""
    global _last_pct
    if total <= 0:
        return
    pct = min(100, count * block_size * 100 // total)
    if pct >= _last_pct + 10:
        _last_pct = pct - (pct % 10)
        print(f"  {pct:3d}%  {count * block_size / 1e6:6.1f} MB")
        sys.stdout.flush()


def download(dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 50_000_000:
        print(f"{dest.name} already downloaded ({dest.stat().st_size/1e6:.0f} MB).")
        return
    for url in MIRRORS:
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dest, reporthook=progress)
            print()
            return
        except Exception as e:  # noqa: BLE001 — try the next mirror
            print(f"\n  failed: {type(e).__name__}: {e}")
    raise SystemExit("All mirrors failed. Download EuroSAT.zip manually into data/.")


def find_class_root(extracted: Path) -> Path:
    """Archive layout varies by mirror ('2750/' or 'EuroSAT/'). Find the real root."""
    candidates = [extracted, *(p for p in extracted.rglob("*") if p.is_dir())]
    for d in candidates:
        subdirs = [p for p in d.iterdir() if p.is_dir()]
        if len(subdirs) == EXPECTED_CLASSES and any(s.glob("*.jpg") for s in subdirs):
            return d
    raise SystemExit(f"Could not locate the {EXPECTED_CLASSES} class folders under {extracted}.")


def main(out_dir: Path) -> None:
    staging = out_dir.parent / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "EuroSAT.zip"

    download(archive)

    extracted = staging / "extracted"
    if not extracted.exists():
        print("Extracting...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)

    images = out_dir / "images"
    if images.exists():
        print(f"{images} already exists — leaving it alone.")
    else:
        src = find_class_root(extracted)
        images.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {src} -> {images}")
        shutil.copytree(src, images)

    classes = sorted(p.name for p in images.iterdir() if p.is_dir())
    total = sum(1 for _ in images.rglob("*.jpg"))

    print(f"\nClasses ({len(classes)}): {classes}")
    print(f"Images: {total}")

    if len(classes) != EXPECTED_CLASSES:
        raise SystemExit(f"Expected {EXPECTED_CLASSES} classes, got {len(classes)}.")
    if total != EXPECTED_IMAGES:
        print(f"WARNING: expected {EXPECTED_IMAGES} images, got {total}.")

    print("\nDone. Next: python scripts/make_splits.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/v1_eurosat_rgb_64"))
    main(ap.parse_args().out)
