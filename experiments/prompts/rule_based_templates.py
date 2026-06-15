"""Rule-based prompt baseline for ablation studies."""
from pathlib import Path
from typing import Iterable

from backend.app.generator import build_rule_based_prompt as build_backend_rule_based_prompt


def build_rule_based_prompt(
    tones: Iterable[str],
    kansei_words: Iterable[str],
    garment_type: str = "fashion garment",
    sketch_path: str | Path | None = None,
) -> str:
    """Build a deterministic baseline prompt through the backend prompt system."""
    metadata = f"Sketch file: {Path(sketch_path).name}" if sketch_path else None
    return build_backend_rule_based_prompt(
        selected_tones=list(tones),
        selected_kansei=list(kansei_words),
        garment_type=garment_type,
        sketch_metadata=metadata,
    )
