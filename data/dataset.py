"""
data/dataset.py
Dataset classes and DataLoader factories for the autoencoder anomaly-detection pipeline.

Expected folder layout
----------------------
<data_root>/
    train/
        0/   ← real faces only (used for AE training)
        1/   ← AI-generated (ignored during training)
    val/
        0/
        1/
    test/
        0/
        1/
"""

from pathlib import Path
from typing import Literal

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ──────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Augmentation pipeline for training (real faces only)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),          # → [0, 1]
    ])


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic pipeline for validation / test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# ──────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────

class RealFacesDataset(Dataset):
    """
    Returns only real-face images (class 0 folder) — **no labels**.
    Used exclusively for autoencoder training.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None):
        self.root      = Path(root) / split / "0"
        self.transform = transform or get_train_transform()
        self.paths     = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No images found in {self.root}. "
                "Check DATA_ROOT and that the '0' sub-folder exists."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class FaceAnomalyDataset(Dataset):
    """
    Returns images from both class folders with integer labels.
        0 → real face
        1 → AI-generated face
    Used for validation and test evaluation.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["val", "test"],
        transform=None,
    ):
        self.root      = Path(root) / split
        self.transform = transform or get_eval_transform()
        self.samples: list[tuple[Path, int]] = []

        for label_str in ("0", "1"):
            folder = self.root / label_str
            if not folder.exists():
                raise FileNotFoundError(
                    f"Expected folder {folder} does not exist."
                )
            label = int(label_str)
            for p in sorted(folder.rglob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((p, label))

        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.root}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────
# DataLoader factories
# ──────────────────────────────────────────────────────────────

def get_train_loader(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
) -> DataLoader:
    """DataLoader for AE training — real images only, no labels."""
    dataset = RealFacesDataset(
        root=data_root,
        split="train",
        transform=get_train_transform(image_size),
    )
    print(f"[Train] {len(dataset):,} real face images found.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_eval_loader(
    data_root: str | Path,
    split: Literal["val", "test"],
    batch_size: int = 64,
    num_workers: int = 2,
    image_size: int = 224,
) -> DataLoader:
    """DataLoader for val / test — both classes with labels."""
    dataset = FaceAnomalyDataset(
        root=data_root,
        split=split,
        transform=get_eval_transform(image_size),
    )
    n_real = sum(1 for _, l in dataset.samples if l == 0)
    n_ai   = sum(1 for _, l in dataset.samples if l == 1)
    print(f"[{split.capitalize()}] {n_real:,} real | {n_ai:,} AI-generated")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )