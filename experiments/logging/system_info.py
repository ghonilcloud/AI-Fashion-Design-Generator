"""System metadata capture for reproducibility logs."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_system_info() -> dict[str, Any]:
    """Collect Python, package, and device metadata without requiring a GPU."""
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": _package_version("torch"),
            "diffusers": _package_version("diffusers"),
            "transformers": _package_version("transformers"),
            "google-genai": _package_version("google-genai"),
            "pillow": _package_version("pillow"),
            "numpy": _package_version("numpy"),
        },
    }

    try:
        import torch

        info["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as exc:
        info["torch"] = {"error": str(exc)}

    return info

