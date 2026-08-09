"""Resumable DDPM training loop.

Free GPU sessions get killed at 12h. Every epoch writes a full checkpoint
(model, EMA, optimizer, scheduler, RNG state) so `--resume` picks up exactly
where it stopped. An 8-hour run will not survive in one session.

    python -m satdiff.train --config configs/v1.yaml
    python -m satdiff.train --config configs/v1.yaml --resume
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers.training_utils import EMAModel
from torchvision.utils import save_image
from tqdm.auto import tqdm

from .data import denormalize, make_loader
from .model import build_model, build_schedulers, null_class_id
from .sample import fixed_seed_grid

def checkpoint_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"


def hub_repo(cfg: dict) -> str | None:
    """Hub repo id, or None if Hub sync is off."""
    h = cfg.get("hub") or {}
    return h.get("repo_id") if h.get("enabled", True) else None


def hub_preflight(cfg: dict) -> None:
    """Prove the token can write BEFORE burning GPU hours.

    hub_push deliberately swallows errors so one flaky upload cannot kill a
    good run — but that means a bad token degrades silently into "no backup at
    all", which is how a finished 100-epoch run gets lost to a wiped session.
    Checking once at startup turns that into an immediate, obvious failure.
    """
    repo = hub_repo(cfg)
    if not repo:
        print("hub sync OFF — checkpoints are local only.")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        who = api.whoami()          # reads HF_TOKEN from the environment
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        print(f"hub sync ON  -> {repo}  (as {who['name']})")
    except Exception as e:
        raise SystemExit(
            f"\nHub preflight FAILED: {e.__class__.__name__}: {e}\n\n"
            f"  Cannot write to {repo}, so checkpoints would not survive this\n"
            f"  session. Refusing to start — fix the token first:\n"
            f"    1. hf.co/settings/tokens -> New token -> type WRITE\n"
            f"    2. Kaggle: Add-ons -> Secrets -> HF_TOKEN -> paste -> Save\n"
            f"    3. Re-run the setup cell so the new value is loaded\n\n"
            f"  To train anyway with no backup, set hub.enabled: false.\n"
        )


def hub_push(cfg: dict, path: Path, epoch: int) -> None:
    """Upload the checkpoint. Never fatal — a failed push must not kill a run
    that is otherwise fine. The local copy is still on disk either way."""
    repo = hub_repo(cfg)
    if not repo:
        return
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(path), path_in_repo="last.pt",
            repo_id=repo, repo_type="model",
            commit_message=f"epoch {epoch + 1}",
        )
        print(f"  pushed epoch {epoch + 1} -> {repo}")
    except Exception as e:
        print(f"  hub push failed ({e.__class__.__name__}: {e}); continuing")


def hub_pull(cfg: dict, path: Path) -> bool:
    """Fetch last.pt from the Hub when it is not on disk. This is what makes
    --resume work on Kaggle, where /kaggle/working is wiped between sessions
    and there is no Drive to fall back on."""
    repo = hub_repo(cfg)
    if not repo or path.exists():
        return path.exists()
    try:
        from huggingface_hub import hf_hub_download
        print(f"no local checkpoint; trying {repo}")
        src = hf_hub_download(repo_id=repo, filename="last.pt", repo_type="model")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, path)
        print(f"pulled checkpoint from {repo}")
        return True
    except Exception as e:
        print(f"hub pull failed ({e.__class__.__name__}: {e})")
        return False


def grids_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["results_dir"]) / "grids"


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(path, epoch, model, ema, opt, sched, scaler, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": opt.state_dict(),
        "lr_scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, tmp)
    tmp.replace(path)  # atomic — a killed session never leaves a half-written file


def main(cfg_path: Path, resume: bool, checkpoint_dir: str | None = None,
         results_dir: str | None = None) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    if checkpoint_dir:
        cfg["paths"]["checkpoint_dir"] = checkpoint_dir
    if results_dir:
        cfg["paths"]["results_dir"] = results_dir
    ckpt = checkpoint_path(cfg)
    grids = grids_dir(cfg)
    chash = config_hash(cfg)
    set_seed(cfg["seed"])
    print(f"checkpoints -> {ckpt.resolve()}")
    print(f"grids       -> {grids.resolve()}")

    # Before anything expensive: if the run cannot be backed up, say so now.
    hub_preflight(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  config={chash}")
    if device.type != "cuda":
        print("WARNING: no GPU. On Kaggle: Settings -> Accelerator -> GPU P100.")

    loader = make_loader(
        cfg["data"]["root"], "train",
        batch_size=cfg["train"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"train batches/epoch: {len(loader)}")

    model = build_model(cfg).to(device)
    ddpm, ddim = build_schedulers(cfg)
    ema = EMAModel(model.parameters(), decay=cfg["train"]["ema_decay"])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    total_steps = len(loader) * cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, cfg["train"]["warmup_steps"], total_steps))

    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if resume:
        hub_pull(cfg, ckpt)
    if resume and ckpt.exists():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["lr_scheduler"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        start_epoch = ck["epoch"] + 1
        print(f"resumed from epoch {ck['epoch']}")
    elif resume:
        print(f"--resume given but {ckpt} not found; starting fresh.")

    null_id = null_class_id(cfg)
    p_drop = cfg["model"]["label_dropout"]
    grids.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        running, t0 = 0.0, time.time()
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg['train']['epochs']}")

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Drop some labels to the null class so CFG works at sample time.
            if p_drop > 0:
                mask = torch.rand(labels.shape, device=device) < p_drop
                labels = torch.where(mask, null_id, labels)

            noise = torch.randn_like(images)
            t = torch.randint(0, ddpm.config.num_train_timesteps,
                              (images.size(0),), device=device).long()
            noisy = ddpm.add_noise(images, noise, t)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(noisy, t, labels).sample
                loss = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.step(model.parameters())

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = running / len(loader)
        secs = time.time() - t0
        left = (cfg["train"]["epochs"] - epoch - 1) * secs / 3600
        print(f"epoch {epoch+1}  loss {avg:.4f}  {secs:.0f}s  ~{left:.1f}h remaining")

        if (epoch + 1) % cfg["train"]["checkpoint_every"] == 0:
            save_checkpoint(ckpt, epoch, model, ema, opt, sched, scaler, cfg)
            # Pushing 1.1 GB takes longer than an epoch takes to train, so this
            # is deliberately less frequent than the local save. Worst case you
            # lose hub_push_every epochs, not the run.
            if (epoch + 1) % cfg["train"].get("hub_push_every", 5) == 0:
                hub_push(cfg, ckpt, epoch)

        if (epoch + 1) % cfg["train"]["grid_every"] == 0:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            grid = fixed_seed_grid(model, ddim, cfg, device)
            save_image(denormalize(grid), grids / f"epoch_{epoch+1:03d}.png", nrow=4)
            ema.restore(model.parameters())
            print(f"  wrote {grids / f'epoch_{epoch+1:03d}.png'}")

    # Always push the final state, whatever hub_push_every last landed on.
    # This is the one that matters: the finished model.
    if ckpt.exists():
        print("\npushing final checkpoint...")
        hub_push(cfg, ckpt, cfg["train"]["epochs"] - 1)

    print("\nTraining done. Next: python -m satdiff.eval --config configs/v1.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Override paths.checkpoint_dir. On Colab point this at Drive.")
    ap.add_argument("--results-dir", default=None,
                    help="Override paths.results_dir. On Colab point this at Drive, "
                         "or the sample grids die with the session.")
    a = ap.parse_args()
    main(a.config, a.resume, a.checkpoint_dir, a.results_dir)
