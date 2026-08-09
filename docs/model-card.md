# Model Card — SatDiff v1

Class-conditional denoising diffusion model that generates 64×64 synthetic
satellite image tiles across the 10 EuroSAT land-use classes.

- **Weights:** `shairaam/satdiff-v1` on the Hugging Face Hub (`last.pt`)
- **Code:** https://github.com/DrKingSchultz69/satdiff-v2
- **License:** MIT, matching the EuroSAT source data
- **Last updated:** 2026-08-10

## What it does

Given one of 10 class labels, it generates a 64×64 RGB tile that looks like a
Sentinel-2 patch of that land-use type. Classes: AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential,
River, SeaLake.

## Intended use

- Augmenting training sets for land-use classifiers when more real Sentinel-2
  data is not obtainable
- Learning and teaching diffusion models end to end

## Out of scope

**Do not** present outputs as real Earth observations. These are synthetic and
depict no actual place. Specifically not suitable for:

- Any factual claim about a real location, at any time
- Disaster, conflict, or infrastructure reporting
- Environmental monitoring, land registry, or insurance assessment
- Anything downstream of "this is what that place looks like"

Generating imagery that purports to show real events is the primary misuse
surface for this class of model. Every output carries a visible watermark and
embedded provenance metadata; do not strip them.

## Architecture

| Component | Value |
|---|---|
| Backbone | `diffusers.UNet2DModel`, 64×64 |
| Base channels | 128, multipliers (1, 2, 2, 4) |
| Layers per block | 2 |
| Attention | at 16×16 resolution |
| Conditioning | sinusoidal timestep embedding + class embedding, summed |
| Class embeddings | 11 = 10 classes + 1 null, for classifier-free guidance |
| Train timesteps | 1000 |
| Beta schedule | cosine (`squaredcos_cap_v2`) — see ADR 0002 |
| Prediction target | epsilon |
| Sampler | DDIM, 50 steps, guidance scale 2.0 |

The model takes `(sample, timestep, class_label)`. The timestep input is not
optional: without it a denoiser cannot know the noise level and can only learn
an average denoiser across all timesteps.

## Training data

EuroSAT RGB — 27,000 Sentinel-2 tiles, 64×64, 10 classes, MIT licensed.
Native resolution, no resizing. See `docs/dataset-card.md`.

Splits are assigned by SHA-256 of each file's relative path, not by directory
listing order, so they are identical on any machine. This is what made it safe
to move training between machines mid-run without leaking val data into train.

| Split | Images |
|---|---|
| train | 21,549 |
| val | 2,686 |
| test | 2,765 |

Augmentation: random horizontal and vertical flips only. Satellite imagery has
no canonical "up", so flips are label-preserving; crops and rotations were not
used.

## Training procedure

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 1e-4 |
| LR schedule | 500-step warmup, then cosine to zero over the full run |
| Batch size | 64 |
| Epochs | 100 (33,600 steps) |
| EMA | 0.9999 (evaluation uses EMA weights) |
| Precision | fp16 mixed |
| Gradient clipping | 1.0 |
| Label dropout | 0.1, to the null class, enabling CFG |
| Hardware | NVIDIA T4, ~12 GPU-hours total |

Batch 64 rather than the 128 in the TRD: 128 peaks at 13.7 GiB on a T4 and
OOMs in the upsample path, where every skip connection is still resident.

## Evaluation

Thresholds were fixed in `docs/eval-plan.md` **before** training started.
The test split was evaluated exactly once, for this card.

| Metric | Val | **Test** | Ship | Good |
|---|---|---|---|---|
| KID ↓ | 0.0098 ± 0.0006 | **0.0103 ± 0.0006** | <0.05 | <0.02 |
| CAS ↑ | 89.0% | **89.4%** | ≥65% | ≥80% |

Both metrics clear the "good" band. Val and test agree closely, so the val
split was not overfit during development.

**KID** (Kernel Inception Distance) over 2,700 generated images, 270 per class,
subset size 1000. KID rather than FID because FID is badly biased at this
sample count and KID is not.

**CAS** (Classification Accuracy Score) is the metric that measures the actual
product feature — whether conditioning works. A ResNet-18 is trained on the
real train split, then asked to classify generated images; CAS is the fraction
assigned to the class they were conditioned on. Random baseline is 10%.

**The classifier must be validated before CAS means anything.** An initial
from-scratch ResNet-18 reached only 81.7% on the real test split, below the
90% validity gate, and reported CAS 68.0%. Switching to ImageNet-pretrained
weights raised the classifier to 97.4% and CAS to 89.0% — with no change
whatsoever to the generator. A weak classifier misreads correctly-conditioned
generated tiles at roughly the rate it misreads real ones, so it understates
CAS. Always check the classifier's real-test accuracy before trusting CAS.

## Limitations

- **64×64 only.** Not photorealistic, and not comparable to real Sentinel-2
  imagery at native detail. Plausible tiles, not real observations.
- **Global color instability.** At epoch 45, structurally correct tiles
  sometimes carried strong pink, purple, or orange casts. Structure is learned
  faster than global color. A likely contributor is non-zero terminal SNR in
  the cosine schedule, which limits how much the model can control overall
  brightness; `rescale_betas_zero_snr` would be the first thing to try.
- **The epoch-100 fixed-seed grid was not inspected.** Sample grids lived only
  in an ephemeral working directory that was wiped. Metrics are strong and KID
  would likely reflect severe mode collapse, but the eye test — which catches
  collapse that metrics miss — has not been re-run at epoch 100. Regenerate it
  from the checkpoint before relying on this model.
- **Class confusion is expected between PermanentCrop and AnnualCrop.** These
  overlap in the source data. Check the confusion matrix before attributing
  such errors to the generator.
- **No per-class CAS breakdown** is recorded; only the aggregate.

## Safety

Every generated image carries:

1. A visible "AI-GENERATED" watermark, bottom-right
2. PNG `tEXt` metadata: model version, class, seed, timestamp

## Reproducing

```bash
git clone https://github.com/DrKingSchultz69/satdiff-v2
pip install -r requirements.txt
python scripts/download_data.py && python scripts/make_splits.py
python -m satdiff.train --config configs/v1.yaml
python -m satdiff.eval  --config configs/v1.yaml --split val
```

Every run is reproducible from `configs/v1.yaml`; its hash is recorded in
`experiments.csv` alongside each eval result.

## Citation

EuroSAT: Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark
for Land Use and Land Cover Classification", IEEE JSTARS, 2019.
