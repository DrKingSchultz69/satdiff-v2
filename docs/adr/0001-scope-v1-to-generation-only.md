# ADR 0001 — Scope v1 to class-conditional generation only

**Date:** 2026-08-06 · **Status:** accepted

**Context.** The previous SatDiff attempt scoped generation, super-resolution, and semantic segmentation into one model. None of the three worked. Super-resolution returned the model's raw noise prediction rather than an image, and segmentation argmaxed over 3 RGB channels so it could only ever output values 0–2 regardless of the 10 classes.

**Decision.** v1 ships class-conditional generation only. Super-resolution and segmentation are removed from scope entirely.

**Consequence.** One conditioning path, one eval metric set, one thing to debug. SR needs a different setup (SR3-style low-res concatenation) and belongs in v2. Diffusion-backbone segmentation is an open research problem and is dropped, not deferred.
