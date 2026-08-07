# ADR 0002 — Cosine beta schedule over linear

**Date:** 2026-08-06 · **Status:** accepted

**Context.** DDPM (Ho et al. 2020) used a linear beta schedule tuned for 256×256. At low resolution it destroys image information too early — by the midpoint of the forward process a 64×64 image is already almost pure noise, so a large share of training timesteps teach the model nothing.

**Decision.** Use the cosine schedule from Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021). `DDPMScheduler(beta_schedule="squaredcos_cap_v2")`.

**Consequence.** Better sample quality at 64×64 for free. If samples come out over-smoothed, this is the first knob to revisit.
