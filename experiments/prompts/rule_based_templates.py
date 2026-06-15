"""Rule-based prompt baseline for ablation studies."""
from pathlib import Path
from typing import Iterable


def build_rule_based_prompt(
    tones: Iterable[str],
    kansei_words: Iterable[str],
    garment_type: str = "fashion garment",
    sketch_path: str | Path | None = None,
) -> str:
    """Build a deterministic baseline prompt without an LLM call."""
    tone_text = ", ".join(t for t in tones if t) or "balanced"
    kansei_text = ", ".join(k for k in kansei_words if k) or "clean"
    sketch_note = "preserving the uploaded sketch silhouette"
    if sketch_path:
        sketch_note = f"preserving the silhouette from {Path(sketch_path).name}"

    return (
        f"Fashion illustration of a {garment_type}, {sketch_note}. "
        f"Style direction: {tone_text}. Kansei attributes: {kansei_text}. "
        "Render as a clean CAD-inspired fashion concept with realistic fabric "
        "texture, clear garment edges, wearable proportions, and a plain neutral background."
    )

