"""Verify a GDRBench split file against an images tree on disk."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LayoutReport:
    total: int = 0
    missing: list[str] = field(default_factory=list)
    label_counts: dict[int, int] = field(default_factory=dict)


def verify_split(split_file, images_root) -> LayoutReport:
    split_file = Path(split_file)
    images_root = Path(images_root)
    rep = LayoutReport()
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rel, label = line.rsplit(" ", 1)
        label = int(label)
        rep.total += 1
        rep.label_counts[label] = rep.label_counts.get(label, 0) + 1
        if not (images_root / rel).exists():
            rep.missing.append(rel)
    return rep
