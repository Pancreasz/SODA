"""GDRBench dataset + loaders for the fresh SODA harness (Kaggle/torch only).

Split semantics match DGDR: source domains use their `_train.txt` for training
and `_crossval.txt` for validation; the target domain uses train+crossval
combined as the held-out test set.
"""
from __future__ import annotations

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_split(split_file, images_root, domain_idx):
    items = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, label = line.rsplit(" ", 1)
            items.append((os.path.join(images_root, rel), int(label), domain_idx))
    return items


class GDRDataset(Dataset):
    def __init__(self, root, domains, split, transform):
        self.transform = transform
        self.items = []
        splits_dir = os.path.join(root, "splits")
        for di, dom in enumerate(domains):
            if split == "test":
                for kind in ("train", "crossval"):
                    self.items += _read_split(
                        os.path.join(splits_dir, f"{dom}_{kind}.txt"), root, di)
            else:
                self.items += _read_split(
                    os.path.join(splits_dir, f"{dom}_{split}.txt"), root, di)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, dom = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, dom


def _train_tf(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_tf(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(root, sources, target, img_size=224, batch_size=16, num_workers=2):
    train_ds = GDRDataset(root, sources, "train", _train_tf(img_size))
    val_ds = GDRDataset(root, sources, "crossval", _eval_tf(img_size))
    test_ds = GDRDataset(root, [target], "test", _eval_tf(img_size))
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)
