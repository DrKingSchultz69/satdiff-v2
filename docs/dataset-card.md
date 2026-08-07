# Dataset Card — EuroSAT RGB

Format follows *Datasheets for Datasets* (Gebru et al., 2018), trimmed to what matters here.

## Identity

| Field | Value |
|---|---|
| Name | EuroSAT (RGB variant) |
| Source | https://github.com/phelber/EuroSAT |
| Paper | Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification" (2019) |
| License | MIT — permissive, commercial use OK |
| Size | 27,000 images, ~90 MB |
| Resolution | 64×64, RGB JPEG |
| Local version tag | `v1_eurosat_rgb_64` |

## Why this one

Checked against alternatives — license is the deciding factor since the app is publicly hosted.

| Rejected | Reason |
|---|---|
| RESISC45 | Research-only license |
| AID, PatternNet | Google Earth derived, redistribution unclear |
| UC Merced | 2,100 images — too small for a generative model |
| BigEarthNet | Good license (CDLA-Permissive-1.0), but 590k images is v2 scale-up |

## Classes

10, roughly balanced (2,000–3,000 each):

AnnualCrop · Forest · HerbaceousVegetation · Highway · Industrial · Pasture · PermanentCrop · Residential · River · SeaLake

## Preprocessing

Deliberately minimal:

1. Use the **RGB JPEG release directly**. Do not download the 13-band `.tif` version and convert — unnecessary step.
2. **Keep native 64×64.** Do not resize down.
3. Normalize to `[-1, 1]` (mean 0.5, std 0.5 per channel).
4. Augment with random horizontal + vertical flip only. Satellite imagery has no canonical "up", so flips are free data. No crops, no color jitter.

## Splits

Stratified by class, seeded, written to `data/v1_eurosat_rgb_64/splits.json`:

- train 80% (21,600)
- val 10% (2,700) — for KID during training
- test 10% (2,700) — touched once, at the end

Split by **file hash**, not `os.listdir()` order.

## Known limitations

- Sentinel-2 only, so European geography dominates. A model trained here will not generalize to e.g. tropical or arctic terrain.
- 64×64 is far below what a real remote-sensing workflow uses.
- Class labels are single-label; real satellite tiles usually contain multiple land types.
- Some classes are visually near-identical at 64×64 (PermanentCrop vs AnnualCrop). Expect conditioning to be weakest there.

## Provenance check

Record in `data/v1_eurosat_rgb_64/manifest.json`: per-file SHA-256, download date, source URL. An experiment log entry is meaningless without this.
