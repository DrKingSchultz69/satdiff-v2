# Eval Plan — SatDiff v1

**Written before training starts.** Thresholds are fixed here so results can't move the goalposts later.

## The three metrics

### 1. KID — image quality

Kernel Inception Distance against the **val** split.

Use KID, not FID. FID is badly biased at small sample counts and you'll be evaluating on ~2,700 images. KID is unbiased at that scale.

- Sample 2,700 generated images (270 per class)
- Compare against val split
- `torchmetrics.image.kid.KernelInceptionDistance`, subset size 1000

| Bar | KID |
|---|---|
| Ship | < 0.05 |
| Good | < 0.02 |

### 2. CAS — does conditioning actually work

Classification Accuracy Score. This is the one that measures the product feature.

1. Train a small classifier (ResNet-18, ~10 min) on the **real** train split
2. Confirm it hits >90% on the real test split — if not, the classifier is broken, not the generator
3. Run it over generated images
4. CAS = % assigned to the class they were conditioned on

| Bar | CAS |
|---|---|
| Random baseline | 10% |
| **Abort checkpoint (epoch 20)** | ≥ 40%, else stop and fix architecture |
| Ship | ≥ 65% |
| Good | ≥ 80% |

Expect PermanentCrop/AnnualCrop confusion. That's the data, not the model — check the confusion matrix before blaming training.

### 3. Fixed-seed grid — the eye test

10×4 grid, one row per class, **same 4 seeds every time**. Rendered every 5 epochs, committed to `results/grids/epoch_NNN.png`.

Metrics miss mode collapse. Your eyes catch it in two seconds. If all four seeds in a row look identical, the model has collapsed regardless of what KID says.

## When to stop training

Stop at whichever comes first:

1. Val KID hasn't improved for 15 epochs
2. Both ship bars met (KID < 0.05 AND CAS ≥ 65%)
3. 8 GPU-hours spent

## Rules

- **Test split is touched exactly once**, for the final numbers in the model card. Everything during development uses val.
- Every eval run appends a row to `experiments.csv`: config hash, dataset tag, epoch, KID, CAS, checkpoint URI.
- Report KID with its standard deviation. A single number without spread is not a result.
