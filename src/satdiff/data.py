"""EuroSAT dataset reading the frozen splits.json.

Preprocessing is deliberately minimal:
  - native 64x64, no resize
  - normalize to [-1, 1]
  - random h/v flip only (satellite imagery has no canonical "up")
"""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class EuroSATSplit(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int = 64,
                 augment: bool | None = None):
        self.root = Path(root)
        meta = json.loads((self.root / "splits.json").read_text())

        self.classes: list[str] = meta["classes"]
        self.items: list[dict] = meta["splits"][split]
        if not self.items:
            raise ValueError(f"Split '{split}' is empty.")

        augment = (split == "train") if augment is None else augment
        ops = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] if augment else []
        self.transform = transforms.Compose([
            *ops,
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = Image.open(self.root / "images" / item["path"]).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        return self.transform(img), item["label"]


def make_loader(root, split, batch_size, image_size=64, num_workers=4, shuffle=None):
    ds = EuroSATSplit(root, split, image_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        persistent_workers=num_workers > 0,
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], clamped."""
    return (x * 0.5 + 0.5).clamp(0, 1)
