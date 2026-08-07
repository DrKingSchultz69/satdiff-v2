# SatDiff

Class-conditional diffusion model generating 64×64 synthetic satellite imagery across 10 EuroSAT land-use classes.

## Run it

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # Windows: $env:PYTHONPATH="src"

python scripts/download_data.py          # ~90 MB, 2 min
python scripts/make_splits.py            # 1 min
python -m satdiff.train --config configs/v1.yaml
python -m satdiff.eval  --config configs/v1.yaml --split val
```

Training killed by a session timeout? Add `--resume`.

## Where things are

| Path | What |
|---|---|
| `docs/prd.md` | Scope, done criteria |
| `docs/eval-plan.md` | KID + CAS thresholds, fixed before training |
| `docs/dataset-card.md` | EuroSAT provenance, license, splits |
| `docs/trd.md` | Stack, model spec, API |
| `docs/adr/` | Why each decision was made |
| `results/grids/` | Fixed-seed sample grids, one per 5 epochs |
| `experiments.csv` | One row per eval run |

## Status

- [x] Docs
- [x] Data pipeline, model, training, eval
- [ ] First training run
- [ ] Gradio app
- [ ] Deployed to HF Spaces

## Data

EuroSAT RGB — 27,000 images, 64×64, 10 classes, MIT licensed. Native resolution, no downsizing.

## Note on outputs

Generated images are synthetic and carry a visible watermark plus embedded metadata. They are not real observations and must not be presented as such.
