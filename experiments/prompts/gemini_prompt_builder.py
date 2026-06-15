"""Gemini prompt wrapper for experiments.

This module deliberately calls the production prompt builder so API runs and
paper experiments share the same prompt-generation behavior.
"""
from pathlib import Path
from typing import Iterable

from backend.app.generator import build_prompt_for_strategy


def generate_gemini_image_prompt(
    tones: Iterable[str],
    kansei_words: Iterable[str],
    sketch_path: str | Path | None = None,
    model: str = "gemini-2.5-flash",
) -> str:
    """Return the final image prompt produced through the backend Gemini path."""
    prompt, _ = build_prompt_for_strategy(
        list(tones),
        list(kansei_words),
        prompt_strategy="llm",
        sketch_path=sketch_path,
        model=model,
    )
    return prompt
