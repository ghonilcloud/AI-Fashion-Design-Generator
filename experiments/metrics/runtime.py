"""Runtime and memory measurement helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def measure_runtime() -> Iterator[dict[str, Any]]:
    """Measure total generation latency and optional CUDA peak VRAM usage."""
    stats: dict[str, Any] = {"started_at": time.time()}
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None

    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats["elapsed_seconds"] = time.perf_counter() - start
        try:
            import torch

            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                stats["cuda_peak_memory_gb"] = peak_bytes / (1024**3)
                stats["cuda_peak_memory_mb"] = peak_bytes / (1024**2)
            else:
                stats["cuda_peak_memory_gb"] = None
                stats["cuda_peak_memory_mb"] = None
        except Exception:
            stats["cuda_peak_memory_gb"] = None
            stats["cuda_peak_memory_mb"] = None


def get_peak_vram_gb() -> float | None:
    """Return peak allocated CUDA VRAM in GB, or None when CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        return None
