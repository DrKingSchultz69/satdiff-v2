"""Is the global-colour drift fixable at sample time, or does it need a retrain?

Epoch 100 produces correct per-class *structure* with unreliable global
*colour*: four Forest samples came out one green and three blue-grey. The
suspected cause is non-zero terminal SNR — with the cosine schedule the final
training timestep is not quite pure noise, so the model never has to learn
global brightness from nothing, yet inference starts from pure noise.

This samples the same class and seeds under several scheduler settings so the
question can be answered for ~30 GPU-seconds instead of a 12-hour retrain.

    python scripts/test_zero_snr.py                    # Forest, 6 seeds
    python scripts/test_zero_snr.py --class-name River --seeds 8

Read the output grid by row. If a row is consistently the right colour, that
setting fixes it and belongs in configs/v1.yaml. If every row still drifts,
sampling cannot fix this and the retrain is the real answer.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from diffusers import DDIMScheduler
from torchvision.utils import save_image

from satdiff.data import denormalize
from satdiff.eval import load_model
from satdiff.sample import sample

# Each variant is (label, scheduler kwargs, guidance scale override).
#
# `trailing` spacing is the one that can work without retraining: it makes the
# last sampling step land on the true final timestep instead of overshooting.
#
# `rescale_betas_zero_snr` is included to size the effect, but expect it to
# look worse here, not better — enforcing zero terminal SNR under
# epsilon-prediction is degenerate (predicting the noise in pure noise is
# information-free). It is the right fix *combined with training*, which is
# exactly the distinction this test exists to establish.
#
# The low-guidance row separates a different hypothesis: classifier-free
# guidance extrapolates away from the unconditional prediction and can push
# colour out of range on its own.
VARIANTS = [
    ("baseline (as shipped)",        dict(),                                        None),
    ("trailing spacing",             dict(timestep_spacing="trailing"),             None),
    ("trailing + zero-SNR betas",    dict(timestep_spacing="trailing",
                                          rescale_betas_zero_snr=True),             None),
    ("baseline, guidance 1.0 (off)", dict(),                                        1.0),
]


def build_ddim(cfg: dict, **extra) -> DDIMScheduler:
    d = cfg["diffusion"]
    return DDIMScheduler(
        num_train_timesteps=d["num_train_timesteps"],
        beta_schedule=d["beta_schedule"],
        prediction_type=d["prediction_type"],
        **extra,
    )


def main(cfg_path: Path, class_name: str, n_seeds: int, out: Path) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU; this will be slow.")

    classes = json.loads((Path(cfg["data"]["root"]) / "splits.json").read_text())["classes"]
    if class_name not in classes:
        raise SystemExit(f"{class_name!r} not in {classes}")
    class_idx = classes.index(class_name)

    model, epoch = load_model(cfg, device)
    seeds = list(range(n_seeds))

    rows = []
    print(f"\nclass={class_name}  epoch={epoch + 1}  seeds={seeds}\n")
    for label, kwargs, gs in VARIANTS:
        ddim = build_ddim(cfg, **kwargs)
        imgs = [sample(model, ddim, cfg, [class_idx], device, seed=s,
                       guidance_scale=gs, progress=False)[0] for s in seeds]
        batch = torch.stack(imgs)

        # Mean RGB is the whole question in one number: Forest should sit green
        # (channel 1 highest). A blue-dominant mean is the failure, quantified.
        rgb = denormalize(batch).mean(dim=(0, 2, 3)).cpu()
        print(f"  {label:<30} mean RGB = "
              f"({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})"
              f"   {'green-dominant' if rgb[1] == rgb.max() else 'NOT green-dominant'}")
        rows.append(batch)

    grid = torch.cat(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(grid), out, nrow=len(seeds))

    print(f"\nwrote {out.resolve()}")
    print("rows, top to bottom:")
    for i, (label, _, _) in enumerate(VARIANTS, 1):
        print(f"  {i}. {label}")
    print("\nA row that is consistently the right colour is the fix. If every "
          "row drifts,\nsampling cannot repair this and the retrain is the "
          "answer:\n  configs/v1.yaml -> diffusion.rescale_betas_zero_snr: true")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    ap.add_argument("--class-name", default="Forest")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("results/zero_snr_test.png"))
    a = ap.parse_args()
    main(a.config, a.class_name, a.seeds, a.out)
