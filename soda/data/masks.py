"""Regenerate fundus field-of-view (FOV) masks from images.

The FOV mask is the circular retina region on a black border. We threshold
intensity, fill interior holes, and keep the largest connected component.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _to_rgb_array(image) -> np.ndarray:
    if hasattr(image, "convert"):  # PIL.Image
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3]


def generate_fov_mask(image, tol: int = 7) -> np.ndarray:
    """Return a uint8 HxW mask (0 or 255) of the fundus FOV."""
    arr = _to_rgb_array(image)
    gray = arr.mean(axis=2)
    fg = gray > tol
    if not fg.any():
        return np.zeros(gray.shape, dtype=np.uint8)
    fg = ndimage.binary_fill_holes(fg)
    labels, n = ndimage.label(fg)
    if n == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return np.where(labels == largest, 255, 0).astype(np.uint8)
