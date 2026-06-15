"""Gemini prompt wrapper for experiments.

This module deliberately calls the production prompt builder so API runs and
paper experiments share the same prompt-generation behavior.
"""
from pathlib import Path
from typing import Iterable

from backend.app.generator import (
    build_gemini_instruction,
    call_gemini_for_prompt,
    refine_for_image_model,
)


def generate_gemini_image_prompt(
    tones: Iterable[str],
    kansei_words: Iterable[str],
    sketch_path: str | Path | None = None,
    model: str = "gemini-2.5-flash",
) -> str:
    """Return the final image prompt produced through the backend Gemini path."""
    instruction = build_gemini_instruction(
        list(tones),
        list(kansei_words),
        sketch_path=sketch_path,
    )
    gemini_text = call_gemini_for_prompt(
        instruction,
        sketch_path=sketch_path,
        model=model,
    )
    return refine_for_image_model(gemini_text)

