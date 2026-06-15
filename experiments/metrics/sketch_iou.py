"""Sketch edge-overlap metric for generated outputs.

Sketch IoU is higher-is-better: larger values indicate stronger edge overlap
between the original CAD sketch and the generated image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale_array(path: str | Path, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Load, grayscale, and resize an image for edge-based comparison."""
    image = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def canny_edge_map(
    path: str | Path,
    size: tuple[int, int] = (512, 512),
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> np.ndarray:
    """Return a binarized Canny edge map.

    OpenCV's Canny implementation is used when available. A small NumPy
    fallback is provided so the metric remains usable in lightweight
    environments.
    """
    gray = load_grayscale_array(path, size=size)
    try:
        import cv2

        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges > 0
    except ImportError:
        return _canny_edge_map_numpy(gray, low_threshold, high_threshold)


def compute_sketch_iou(
    sketch_path: str | Path,
    generated_path: str | Path,
    canny_low_threshold: int = 100,
    canny_high_threshold: int = 200,
    size: tuple[int, int] = (512, 512),
) -> float:
    """Compute IoU between binarized Canny edge maps of sketch and output."""
    sketch_edges = canny_edge_map(
        sketch_path,
        size=size,
        low_threshold=canny_low_threshold,
        high_threshold=canny_high_threshold,
    )
    generated_edges = canny_edge_map(
        generated_path,
        size=size,
        low_threshold=canny_low_threshold,
        high_threshold=canny_high_threshold,
    )

    intersection = np.logical_and(sketch_edges, generated_edges).sum()
    union = np.logical_or(sketch_edges, generated_edges).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _canny_edge_map_numpy(
    gray: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> np.ndarray:
    blurred = _gaussian_blur(gray.astype(np.float32))
    magnitude, direction = _sobel_gradients(blurred)
    suppressed = _non_maximum_suppression(magnitude, direction)
    return _hysteresis_threshold(suppressed, low_threshold, high_threshold)


def _gaussian_blur(image: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=np.float32,
    )
    kernel /= kernel.sum()
    return _convolve2d(image, kernel)


def _sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = _convolve2d(image, sobel_x)
    gy = _convolve2d(image, sobel_y)
    magnitude = np.hypot(gx, gy)
    direction = np.rad2deg(np.arctan2(gy, gx)) % 180
    return magnitude, direction


def _non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    height, width = magnitude.shape
    output = np.zeros_like(magnitude)
    angle = np.zeros_like(direction)
    angle[((0 <= direction) & (direction < 22.5)) | ((157.5 <= direction) & (direction <= 180))] = 0
    angle[(22.5 <= direction) & (direction < 67.5)] = 45
    angle[(67.5 <= direction) & (direction < 112.5)] = 90
    angle[(112.5 <= direction) & (direction < 157.5)] = 135

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            current_angle = angle[row, col]
            if current_angle == 0:
                before, after = magnitude[row, col - 1], magnitude[row, col + 1]
            elif current_angle == 45:
                before, after = magnitude[row - 1, col + 1], magnitude[row + 1, col - 1]
            elif current_angle == 90:
                before, after = magnitude[row - 1, col], magnitude[row + 1, col]
            else:
                before, after = magnitude[row - 1, col - 1], magnitude[row + 1, col + 1]

            if magnitude[row, col] >= before and magnitude[row, col] >= after:
                output[row, col] = magnitude[row, col]
    return output


def _hysteresis_threshold(
    image: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> np.ndarray:
    strong = image >= high_threshold
    weak = (image >= low_threshold) & ~strong
    edges = strong.copy()

    changed = True
    while changed:
        previous = edges.copy()
        neighbors = _dilate(edges)
        edges = strong | (weak & neighbors)
        changed = not np.array_equal(previous, edges)
    return edges


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    for row_offset in range(3):
        for col_offset in range(3):
            output |= padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
    return output


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            region = padded[row : row + kernel.shape[0], col : col + kernel.shape[1]]
            output[row, col] = float(np.sum(region * kernel))
    return output
