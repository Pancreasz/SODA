"""Grade-preserving fundus style augmentation for L_consist (Kaggle/torch only).

Produces a style-perturbed view x_style whose DR grade is unchanged but whose
camera/style-look differs (brightness/contrast/hue + a mild optical perturbation).
Severity read from x_style must match that of x (that is what L_consist enforces).
"""
from __future__ import annotations

import random
from torchvision import transforms
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _RandomGaussianBlur:
    def __init__(self, p=0.5, radius=(0.1, 1.5)):
        self.p = p
        self.radius = radius

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(*self.radius)
            return TF.gaussian_blur(img, kernel_size=5, sigma=sigma)
        return img


def style_transform(img_size):
    """Grade-preserving style perturbation, then resize + normalize like eval."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        _RandomGaussianBlur(p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
