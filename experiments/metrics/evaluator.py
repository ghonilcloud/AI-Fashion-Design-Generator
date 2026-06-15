"""Unified metric evaluation for generated fashion CAD-sketch outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .clip_score import compute_clip_score
from .lpips_distance import compute_lpips_distance
from .runtime import get_peak_vram_gb
from .sketch_iou import compute_sketch_iou


DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_LPIPS_BACKBONE = "alex"
DEFAULT_CANNY_LOW_THRESHOLD = 100
DEFAULT_CANNY_HIGH_THRESHOLD = 200


def evaluate_generation(
    sketch_path: str | Path,
    generated_image_path: str | Path,
    prompt: str,
    metrics_config: dict[str, Any] | list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Evaluate one generated image.

    Metric interpretation:
    - CLIP text-image similarity is higher-is-better.
    - LPIPS perceptual distance is lower-is-better.
    - Sketch IoU is higher-is-better.
    """
    config = _normalize_metrics_config(metrics_config)
    clip_model_name = config.get("clip_model_name", DEFAULT_CLIP_MODEL_NAME)
    lpips_backbone = config.get("lpips_backbone", DEFAULT_LPIPS_BACKBONE)
    canny_low_threshold = int(config.get("canny_low_threshold", DEFAULT_CANNY_LOW_THRESHOLD))
    canny_high_threshold = int(config.get("canny_high_threshold", DEFAULT_CANNY_HIGH_THRESHOLD))

    results: dict[str, Any] = {
        "clip_score": None,
        "lpips_score": None,
        "sketch_iou": None,
        "latency_seconds": config.get("latency_seconds"),
        "peak_vram_gb": config.get("peak_vram_gb", get_peak_vram_gb()),
        "canny_low_threshold": canny_low_threshold,
        "canny_high_threshold": canny_high_threshold,
        "clip_model_name": clip_model_name,
        "lpips_backbone": lpips_backbone,
        "metric_errors": {},
    }

    if config.get("compute_clip", True):
        try:
            results["clip_score"] = compute_clip_score(
                prompt=prompt,
                image_path=generated_image_path,
                model_name=clip_model_name,
            )
        except Exception as exc:
            results["metric_errors"]["clip_score"] = str(exc)

    if config.get("compute_lpips", True):
        try:
            results["lpips_score"] = compute_lpips_distance(
                sketch_path=sketch_path,
                generated_path=generated_image_path,
                network=lpips_backbone,
            )
        except Exception as exc:
            results["metric_errors"]["lpips_score"] = str(exc)

    if config.get("compute_sketch_iou", True):
        try:
            results["sketch_iou"] = compute_sketch_iou(
                sketch_path=sketch_path,
                generated_path=generated_image_path,
                canny_low_threshold=canny_low_threshold,
                canny_high_threshold=canny_high_threshold,
            )
        except Exception as exc:
            results["metric_errors"]["sketch_iou"] = str(exc)

    return results


def _normalize_metrics_config(
    metrics_config: dict[str, Any] | list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    if metrics_config is None:
        return {}
    if isinstance(metrics_config, dict):
        return metrics_config.copy()

    requested = set(metrics_config)
    return {
        "compute_clip": "clip" in requested or "clip_score" in requested,
        "compute_lpips": "lpips" in requested or "lpips_score" in requested,
        "compute_sketch_iou": "sketch_iou" in requested,
    }

