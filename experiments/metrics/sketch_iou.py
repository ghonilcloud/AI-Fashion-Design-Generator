"""Sketch edge-overlap metric for generated outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _edge_map(path: str | Path, size: tuple[int, int] = (512, 512), threshold: float = 24.0) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32)
    dx = np.zeros_like(arr)
    dy = np.zeros_like(arr)
    dx[:, 1:] = np.abs(arr[:, 1:] - arr[:, :-1])
    dy[1:, :] = np.abs(arr[1:, :] - arr[:-1, :])
    return (dx + dy) >= threshold


def compute_sketch_iou(
    sketch_path: str | Path,
    generated_path: str | Path,
    threshold: float = 24.0,
) -> float:
    """Compute IoU between simple binary edge maps of sketch and output."""
    sketch_edges = _edge_map(sketch_path, threshold=threshold)
    generated_edges = _edge_map(generated_path, threshold=threshold)

    intersection = np.logical_and(sketch_edges, generated_edges).sum()
    union = np.logical_or(sketch_edges, generated_edges).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

