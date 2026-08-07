"""Class-conditional UNet + DDPM/DDIM schedulers.

THE critical detail: the model takes (sample, timestep, class_labels).
A denoiser with no timestep input cannot work — without knowing the noise
level it can only learn an average denoiser across all timesteps. That is
what broke the previous attempt.
"""
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel


def build_model(cfg: dict) -> UNet2DModel:
    m, d = cfg["model"], cfg["data"]
    base = m["base_channels"]
    mults = m["block_mults"]
    size = d["image_size"]

    channels = tuple(base * k for k in mults)
    down, up = [], []
    for i in range(len(mults)):
        res = size // (2 ** i)
        attn = res == m["attention_resolution"]
        down.append("AttnDownBlock2D" if attn else "DownBlock2D")
        up.insert(0, "AttnUpBlock2D" if attn else "UpBlock2D")

    return UNet2DModel(
        sample_size=size,
        in_channels=3,
        out_channels=3,
        layers_per_block=m["layers_per_block"],
        block_out_channels=channels,
        down_block_types=tuple(down),
        up_block_types=tuple(up),
        # +1 for the null class used by classifier-free guidance
        num_class_embeds=m["num_classes"] + 1,
    )


def build_schedulers(cfg: dict) -> tuple[DDPMScheduler, DDIMScheduler]:
    d = cfg["diffusion"]
    kwargs = dict(
        num_train_timesteps=d["num_train_timesteps"],
        beta_schedule=d["beta_schedule"],
        prediction_type=d["prediction_type"],
    )
    return DDPMScheduler(**kwargs), DDIMScheduler(**kwargs)


def null_class_id(cfg: dict) -> int:
    """Index of the 'no class' embedding — the unconditional branch for CFG."""
    return cfg["model"]["num_classes"]
