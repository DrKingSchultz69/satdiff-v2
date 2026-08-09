"""KID + CAS, per docs/eval-plan.md.

    python -m satdiff.eval --config configs/v1.yaml --split val

Thresholds (fixed before training, do not move them):
    KID  ship < 0.05   good < 0.02
    CAS  abort-check >= 40% at epoch 20   ship >= 65%   good >= 80%
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from diffusers.training_utils import EMAModel
from torch import nn
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import models
from tqdm.auto import tqdm

from .data import denormalize, make_loader
from .model import build_model, build_schedulers
from .sample import sample
from .train import hub_pull

def ckpt_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"])


def log_path(cfg: dict) -> Path:
    """The durable record of every eval run. Lives under results_dir so that
    pointing that at Drive keeps it across Colab sessions."""
    return Path(cfg["paths"]["results_dir"]) / "experiments.csv"


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    return (denormalize(x) * 255).to(torch.uint8)


def load_model(cfg, device, use_ema=True):
    # Same Hub fallback as training. Eval usually runs in a fresh session after
    # the one that trained has been wiped, so "no local checkpoint" is the
    # normal case here, not the exceptional one.
    path = ckpt_dir(cfg) / "last.pt"
    if not path.exists():
        hub_pull(cfg, path)
    if not path.exists():
        raise SystemExit(
            f"No checkpoint at {path} and none on the Hub.\n"
            f"  Train first, or point --checkpoint-dir at one that exists."
        )
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    if use_ema:
        ema = EMAModel(model.parameters(), decay=cfg["train"]["ema_decay"])
        ema.load_state_dict(ck["ema"])
        ema.copy_to(model.parameters())
    model.eval()
    return model, ck["epoch"]


# ---------------------------------------------------------------- classifier

def train_classifier(cfg, device, epochs=15):
    """ResNet-18 on real data. Must hit >90% on real test or CAS is meaningless.

    ImageNet-pretrained, not from scratch. From scratch at 8 epochs this reached
    only 81.7% on real test, which put CAS below its own validity threshold and
    made the generator look worse than it is: a classifier that misreads 18% of
    real tiles misreads correctly-conditioned generated ones at a similar rate.
    """
    num_classes = cfg["model"]["num_classes"]
    net = models.resnet18(weights="IMAGENET1K_V1")
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    net = net.to(device)
    train = make_loader(cfg["data"]["root"], "train", 128,
                        num_workers=cfg["data"]["num_workers"])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

    for ep in range(epochs):
        net.train()
        for x, y in tqdm(train, desc=f"classifier {ep+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    acc = classifier_accuracy(net, cfg, device, "test")
    print(f"classifier real-test accuracy: {acc:.1%}")
    if acc < 0.90:
        print("WARNING: <90%. The classifier is broken, not the generator. "
              "Fix this before trusting CAS.")

    cls_ckpt = ckpt_dir(cfg) / "classifier.pt"
    cls_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "real_test_acc": acc}, cls_ckpt)
    return net


@torch.no_grad()
def classifier_accuracy(net, cfg, device, split):
    net.eval()
    loader = make_loader(cfg["data"]["root"], split, 256,
                         num_workers=cfg["data"]["num_workers"], shuffle=False)
    correct = total = 0
    for x, y in loader:
        pred = net(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def get_classifier(cfg, device):
    num_classes = cfg["model"]["num_classes"]
    cls_ckpt = ckpt_dir(cfg) / "classifier.pt"
    if cls_ckpt.exists():
        ck = torch.load(cls_ckpt, map_location=device, weights_only=False)
        # Never reuse a cached classifier that failed its own validity bar —
        # otherwise one bad training run silently poisons every later CAS.
        if ck["real_test_acc"] >= 0.90:
            net = models.resnet18(weights=None, num_classes=num_classes).to(device)
            net.load_state_dict(ck["model"])
            print(f"loaded classifier (real-test acc {ck['real_test_acc']:.1%})")
            return net.eval()
        print(f"cached classifier is only {ck['real_test_acc']:.1%} on real test "
              f"(<90%); retraining rather than trusting it.")
    return train_classifier(cfg, device)


# --------------------------------------------------------------------- eval

def run(cfg, device, split, batch=64):
    model, epoch = load_model(cfg, device)
    _, ddim = build_schedulers(cfg)
    classes = json.loads((Path(cfg["data"]["root"]) / "splits.json").read_text())["classes"]
    n_per = cfg["eval"]["num_samples_per_class"]

    kid = KernelInceptionDistance(subset_size=cfg["eval"]["kid_subset_size"],
                                  normalize=False).to(device)
    # Not under no_grad: on the first run this trains the classifier, which needs
    # backward. Everything after it is inference and takes its own no_grad.
    net = get_classifier(cfg, device)

    with torch.no_grad():
        for x, _ in tqdm(make_loader(cfg["data"]["root"], split, 256,
                                     num_workers=cfg["data"]["num_workers"], shuffle=False),
                         desc=f"real ({split})"):
            kid.update(to_uint8(x).to(device), real=True)

        hits = total = 0
        for cls_idx, cls_name in enumerate(classes):
            remaining = n_per
            pbar = tqdm(total=n_per, desc=f"gen {cls_name}")
            while remaining > 0:
                n = min(batch, remaining)
                imgs = sample(model, ddim, cfg, [cls_idx] * n, device, progress=False)
                kid.update(to_uint8(imgs), real=False)
                hits += (net(imgs).argmax(1) == cls_idx).sum().item()
                total += n
                remaining -= n
                pbar.update(n)
            pbar.close()

    kid_mean, kid_std = kid.compute()
    cas = hits / total

    print(f"\nepoch {epoch+1}   split={split}")
    print(f"KID  {kid_mean:.4f} +/- {kid_std:.4f}   (ship <0.05, good <0.02)")
    print(f"CAS  {cas:.1%}                    (ship >=65%, good >=80%)")
    if cas < 0.40:
        print("CAS below the 40% abort bar — conditioning is not learning. "
              "Stop and fix the architecture before spending more GPU time.")

    log = log_path(cfg)
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "split", "kid_mean", "kid_std", "cas", "checkpoint"])
        w.writerow([epoch + 1, split, f"{kid_mean:.6f}", f"{kid_std:.6f}",
                    f"{cas:.4f}", str(ckpt_dir(cfg) / "last.pt")])
    print(f"appended to {log.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="test is touched ONCE, for the model card")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--results-dir", default=None,
                    help="Where experiments.csv lands. On Colab point this at Drive.")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if a.checkpoint_dir:
        cfg["paths"]["checkpoint_dir"] = a.checkpoint_dir
    if a.results_dir:
        cfg["paths"]["results_dir"] = a.results_dir
    run(cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"), a.split)
