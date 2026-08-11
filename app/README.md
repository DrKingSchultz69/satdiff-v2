---
title: SatDiff
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
short_description: Synthetic 64x64 satellite tiles, 10 EuroSAT classes
---

# SatDiff

Class-conditional diffusion model generating synthetic 64×64 satellite imagery
across 10 EuroSAT land-use classes.

**Outputs are synthetic.** They are not real Earth observations and depict no
actual location. Every image carries a visible watermark and PNG `tEXt`
provenance metadata.

- Weights: [`shairaam/satdiff-v1`](https://huggingface.co/shairaam/satdiff-v1)
- Code and docs: [DrKingSchultz69/satdiff-v2](https://github.com/DrKingSchultz69/satdiff-v2)
- Test-split metrics: KID 0.0103 ± 0.0006, CAS 89.4%

## Deploying

This directory is the Space. The frontmatter above is what HF reads to build it.

`sdk_version` is deliberately omitted so HF installs the gradio version its own
image is built around. Pinning it caused three separate failures — gradio 4.44
against huggingface_hub 1.0 (`HfFolder` gone), against pydantic 2.10 (bool
`additionalProperties`), and gradio 5.0.1 against a newer starlette
(`TemplateResponse` argument order). Each pin fixed one layer and broke the next.

1. Create a Space: SDK **Gradio**, hardware **ZeroGPU** (free for public Spaces)
2. Push `app.py`, `requirements.txt`, and this README to the Space repo
3. The weights repo must be **public**, or the Space needs an `HF_TOKEN` secret
   with read access — otherwise `hf_hub_download` returns 401 at startup

The `@spaces.GPU` decorator on `generate` is what allocates ZeroGPU per request.
The model loads on CPU at import and moves to GPU inside that call: ZeroGPU has
no CUDA context at startup, so initialising on GPU at import time fails.
