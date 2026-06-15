"""LPIPS perceptual distance metric.

LPIPS distance is lower-is-better: smaller values indicate stronger perceptual
similarity between the original sketch and generated image.

Install the optional `lpips` package before enabling this metric.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def compute_lpips_distance(
    sketch_path: str | Path,
    generated_path: str | Path,
    network: str = "alex",
) -> float:
    """Compute LPIPS distance between the sketch and generated output."""
    try:
        import lpips
        import torch
        import torchvision.transforms as transforms
    except Exception as exc:
        raise RuntimeError("LPIPS metric requires the optional `lpips` package.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net=network).to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    sketch = transform(Image.open(sketch_path).convert("RGB")).unsqueeze(0).to(device)
    generated = transform(Image.open(generated_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        return float(loss_fn(sketch, generated).item())
