# PRD — SatDiff v1

**Status:** draft · **Owner:** Shai · **Last updated:** 2026-08-06

## One sentence

A web app that generates synthetic 64×64 satellite images on demand, conditioned on one of 10 EuroSAT land-use classes.

## Who it's for

1. **ML practitioners** who need extra training images for a land-use classifier and can't get more real Sentinel-2 data.
2. **Me**, learning diffusion models end to end and shipping something public.

## v1 scope — IN

- Pick 1 of 10 classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)
- Generate 1–8 images at 64×64
- Download as PNG
- Every output carries a visible watermark + embedded "AI-generated" metadata

## v1 scope — OUT

Explicitly not building these in v1. Each is a separate product.

- ❌ Super-resolution (different conditioning setup, SR3-style)
- ❌ Semantic segmentation (open research problem, not a feature)
- ❌ Resolutions above 64×64
- ❌ User accounts, saved history, payments
- ❌ Custom / user-uploaded classes
- ❌ Text prompts

## Done means

All five true:

1. Public URL anyone can open, no login
2. Class conditioning works — see `eval-plan.md` for the CAS bar
3. Generation returns in under 30s
4. Model card + dataset card published
5. Repo is reproducible: clone → run one command → train

## Non-goals for quality

v1 images will look like plausible 64×64 satellite tiles, not photorealistic imagery. That is acceptable. Photorealism is a v3 problem.

## Risks

| Risk | Mitigation |
|---|---|
| Model never learns conditioning | CAS check at epoch 20; if <40%, fix architecture before continuing |
| Free GPU quota runs out mid-training | Checkpoint every epoch to HF Hub; training must be resumable |
| Misuse (fake disaster imagery) | Watermark + metadata + model card limitations section |

## Rough timeline

| Phase | Time |
|---|---|
| Docs + repo skeleton | 1 day |
| DDPM from scratch (learning) | 3–4 days |
| Production model on `diffusers` | 2 days build, 3–8 GPU-hours train |
| Gradio app + deploy | 1 day |
