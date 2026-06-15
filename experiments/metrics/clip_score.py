"""CLIP text-image semantic alignment metric.

This module is intentionally optional because the model weights may need to be
downloaded before use.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def compute_clip_score(
    prompt: str,
    image_path: str | Path,
    model_name: str = "openai/clip-vit-base-patch32",
) -> float:
    """Compute cosine similarity between prompt and generated image."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("CLIP metric requires torch and transformers.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        score = (text_embeds * image_embeds).sum(dim=-1).item()

    return float(score)

