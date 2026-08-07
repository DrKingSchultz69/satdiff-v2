"""DDIM sampling with classifier-free guidance."""
import torch
from tqdm.auto import tqdm

from .model import null_class_id


@torch.no_grad()
def sample(model, ddim, cfg, class_ids, device, seed=None, guidance_scale=None,
           num_inference_steps=None, progress=True):
    """Generate one image per entry in `class_ids`. Returns [-1, 1] tensor."""
    model.eval()
    n = len(class_ids)
    size = cfg["data"]["image_size"]
    gs = cfg["sample"]["guidance_scale"] if guidance_scale is None else guidance_scale
    steps = cfg["sample"]["num_inference_steps"] if num_inference_steps is None else num_inference_steps

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    x = torch.randn(n, 3, size, size, device=device, generator=gen)
    labels = torch.as_tensor(class_ids, device=device, dtype=torch.long)
    null = torch.full_like(labels, null_class_id(cfg))

    ddim.set_timesteps(steps, device=device)
    iterator = tqdm(ddim.timesteps, desc="sampling", disable=not progress, leave=False)

    for t in iterator:
        if gs > 1.0:
            # One batched forward for cond + uncond instead of two passes.
            both = model(torch.cat([x, x]), t, torch.cat([labels, null])).sample
            cond, uncond = both.chunk(2)
            eps = uncond + gs * (cond - uncond)
        else:
            eps = model(x, t, labels).sample
        x = ddim.step(eps, t, x, generator=gen).prev_sample

    return x


@torch.no_grad()
def fixed_seed_grid(model, ddim, cfg, device, seeds=(0, 1, 2, 3)):
    """One row per class, same seeds every epoch.

    Metrics miss mode collapse. This catches it in two seconds: if all seeds
    in a row look identical, the model collapsed regardless of KID.
    """
    num_classes = cfg["model"]["num_classes"]
    rows = []
    for s in seeds:
        rows.append(sample(model, ddim, cfg, list(range(num_classes)),
                           device, seed=s, progress=False))
    # interleave so each row is one class across all seeds
    stacked = torch.stack(rows, dim=1)              # [C, S, 3, H, W]
    return stacked.reshape(-1, *stacked.shape[2:])  # [C*S, 3, H, W]
