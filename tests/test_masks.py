import numpy as np
from soda.data.masks import generate_fov_mask


def _disk_image(size=100, radius=30):
    yy, xx = np.ogrid[:size, :size]
    c = size // 2
    disk = (xx - c) ** 2 + (yy - c) ** 2 <= radius ** 2
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[disk] = 200
    return img


def test_mask_shape_and_dtype():
    m = generate_fov_mask(_disk_image())
    assert m.shape == (100, 100)
    assert m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, 255})


def test_mask_covers_fundus_not_corners():
    m = generate_fov_mask(_disk_image(size=100, radius=30))
    assert m[50, 50] == 255      # center is fundus
    assert m[0, 0] == 0          # corner is background
    frac = (m > 0).mean()
    assert 0.20 < frac < 0.35    # ~pi*30^2/100^2 = 0.283


def test_all_black_image_returns_empty_mask():
    black = np.zeros((40, 40, 3), dtype=np.uint8)
    m = generate_fov_mask(black)
    assert m.max() == 0
