from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from .prompts import CATEGORIES

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    split: str


@dataclass
class CategoryDataset:
    category: str
    root: Path
    references: list[Path]
    queries: list[Sample]


def _images(path):
    if not path.is_dir():
        raise FileNotFoundError(f"Required LOCO directory is missing: {path}")
    files = sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.name)
    if not files:
        raise ValueError(f"No supported images in {path}")
    return files


def discover_category(data_root: Path, category: str, max_images: int | None = None):
    if category not in CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")
    root = Path(data_root).expanduser().resolve()
    candidates = [root / category, root / "mvtec_loco_anomaly_detection" / category,
                  root / "original" / category, root / "MVTec_Loco" / "original" / category]
    if root.name == category:
        candidates.insert(0, root)
    found = list(dict.fromkeys(p for p in candidates if (p / "train/good").is_dir()))
    if len(found) != 1:
        raise ValueError(f"Expected one LOCO location for {category} beneath {root}; found {len(found)}. "
                         "Set --data-root to the dataset root or category directory explicitly.")
    root = found[0]
    references = _images(root / "train/good")
    good = [Sample(p, 0, "good") for p in _images(root / "test/good")]
    anomalous = [Sample(p, 1, "logical_anomalies") for p in _images(root / "test/logical_anomalies")]
    # Interleave for a useful small smoke run containing both labels. Full runs
    # still use every image; labels never enter any feature-generation stage.
    queries = [sample for pair in zip_longest(good, anomalous) for sample in pair if sample is not None]
    if max_images is not None:
        queries = queries[:max_images]
    return CategoryDataset(category, root, references, queries)


def select_references(references, *, num_runs=1, seed=42, reference_index=None):
    if num_runs < 1 or num_runs > len(references):
        raise ValueError(f"num_runs must be in [1, {len(references)}] for distinct normal references")
    indices = list(range(len(references)))
    if reference_index is not None:
        if reference_index not in indices:
            raise ValueError(f"reference_index must be in [0, {len(references) - 1}]")
        indices.remove(reference_index)
    random.Random(seed).shuffle(indices)
    if reference_index is not None:
        indices.insert(0, reference_index)
    return [references[i] for i in indices[:num_runs]]
