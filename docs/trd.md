# TRD — SatDiff v1

Living doc. Update as the code lands.

## System shape

Training and serving are **separate**. Never serve inference from a Colab/Kaggle notebook — the session dies every 12 hours.

```
Kaggle Notebook (P100, 30 GPU-h/week free)
        │  train
        ▼
Hugging Face Hub  ──  checkpoints + weights (git-lfs, versioned)
        │
        ▼
HF Spaces + ZeroGPU  ──  Gradio app, free shared A100
        │
        ▼
        public URL
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| Training GPU | Kaggle Notebooks | 30 h/week **guaranteed**; Colab free guarantees nothing |
| Diffusion lib | `diffusers` (`UNet2DModel` + `DDPMScheduler`) | Maintained. Hand-rolled version stays in `notebooks/` for learning only |
| Weights hosting | HF Hub | Free, versioned. Stop committing `.pth` to the app repo |
| Inference | HF Spaces ZeroGPU | Free shared A100 for public Spaces |
| UI | Gradio | v1 in an afternoon. Next.js is v2 |
| Config | Hydra or plain YAML | Every run reproducible from a config hash |
| Tracking | Weights & Biases (free tier) | Plus `experiments.csv` as the durable record |

## Model spec

| Param | Value |
|---|---|
| Architecture | `UNet2DModel`, 64×64, base 128 ch, mults (1,2,2,4) |
| Attention | At 16×16 resolution |
| Conditioning | Sinusoidal timestep embedding + class embedding, **added together** |
| Timesteps | 1000 train |
| Schedule | Cosine (better than linear at low res — see ADR 0002) |
| Sampler | DDIM, 50 steps at inference |
| Optimizer | AdamW, lr 1e-4, cosine warmup 500 steps |
| Batch | 64 (T4 15GB). 128 OOMs on a T4 — measured, not estimated |
| EMA | 0.9999 — non-optional, samples are much worse without it |
| Precision | fp16 mixed |

**The critical detail:** the model MUST take `(x, timestep, class_label)`. A denoiser without timestep input cannot work — it can only learn an average over all noise levels. This is what broke the previous attempt.

## Training must be resumable

Free GPU sessions get killed. Every epoch:

1. Save `{model, ema, optimizer, scheduler, epoch, rng_state}`
2. Push to HF Hub
3. On start, look for a checkpoint and resume

Non-negotiable — an 8-hour run will not survive in one session.

## API contract

`POST /generate`

```json
{ "class_name": "Forest", "num_images": 4, "seed": null }
```

```json
{ "job_id": "abc123", "status": "queued" }
```

`GET /jobs/{job_id}` → `{ "status": "done", "images": ["data:image/png;base64,..."] }`

Async, not blocking. 50 DDIM steps takes 3–8s; a blocking request falls over with two concurrent users.

## Output safety

Every generated image gets:

1. Visible watermark, bottom-right, "AI-GENERATED"
2. PNG `tEXt` metadata: model version, class, seed, timestamp

Synthetic satellite imagery has a real misuse surface (fabricated disaster or infrastructure imagery). Both measures cost a few lines.

## Repo layout

```
docs/           prd, trd, eval-plan, dataset-card, model-card, adr/
notebooks/      01_ddpm_from_scratch.ipynb  (learning artifact)
src/satdiff/    data/ models/ diffusion/ train/ sample/ eval/
configs/        *.yaml
app/            Gradio, deploys to HF Spaces
scripts/        download_data.py, make_splits.py
experiments.csv
```
