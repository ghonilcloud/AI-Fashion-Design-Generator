"""Runtime and memory measurement helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def measure_runtime() -> Iterator[dict[str, Any]]:
    """Measure wall-clock seconds and optional CUDA peak memory."""
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
                stats["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            stats["cuda_peak_memory_mb"] = None

