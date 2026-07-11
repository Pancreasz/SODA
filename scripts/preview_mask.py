"""Print coverage stats for a handful of regenerated masks."""
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("D:/fundus/GDRBench/FundusDG_mini/masks")
sample = list(root.rglob("*.png"))[:20]
for p in sample:
    m = np.asarray(Image.open(p))
    print(f"{p.name}: shape={m.shape} coverage={(m>0).mean():.2f}")
