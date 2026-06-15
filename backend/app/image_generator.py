"""
Image generation using Stable Diffusion + ControlNet for CAD-style fashion sketches.

Takes:
- CAD sketch image path
- Gemini-generated prompt
- Optional style parameters

Outputs:
- Generated fashion design image
"""

import os
import logging
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import LineartDetector


logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "sd_v1_5": {
        "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "controlnet_model": "lllyasviel/sd-controlnet-canny",
        "pipeline": "sd",
        "image_size": 512,
    },
    "sdxl": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
        "pipeline": "sdxl",
        "image_size": 1024,
    },
    "ssd_1b": {
        "base_model": "segmind/SSD-1B",
        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
        "pipeline": "sdxl",
        "image_size": 1024,
    },
}

_PIPELINE_CACHE = {}


def resolve_model_config(
    model_key: str = "sd_v1_5",
    base_model: str | None = None,
    controlnet_model: str | None = None,
) -> dict[str, Any]:
    """Resolve registry defaults plus optional explicit model overrides."""
    if model_key not in MODEL_REGISTRY:
        allowed = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_key '{model_key}'. Use one of: {allowed}.")

    model_config = MODEL_REGISTRY[model_key].copy()
    if base_model:
        model_config["base_model"] = base_model
    if controlnet_model:
        model_config["controlnet_model"] = controlnet_model
    model_config["model_key"] = model_key
    return model_config


def resolve_device(device: str | None = None) -> str:
    """Resolve requested device or pick CUDA when available."""
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype: str | torch.dtype | None = None, device: str | None = None) -> torch.dtype:
    """Resolve dtype strings used in JSON configs into torch dtypes."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype:
        normalized = str(dtype).replace("torch.", "").lower()
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if normalized not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype}'.")
        return dtype_map[normalized]
    return torch.float16 if resolve_device(device) == "cuda" else torch.float32


def format_generation_value(value: Any) -> str:
    """Format numeric generation values for stable filenames."""
    if value is None:
        return "none"
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value).replace(".", "p")


def build_generation_filename(
    sketch_id: str,
    model_key: str,
    prompt_strategy: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int | None,
    suffix: str = ".png",
) -> str:
    """Build a filename that encodes core reproducibility parameters."""
    safe_sketch_id = _safe_filename_part(sketch_id)
    safe_model_key = _safe_filename_part(model_key)
    safe_prompt_strategy = _safe_filename_part(prompt_strategy)
    return (
        f"{safe_sketch_id}_{safe_model_key}_{safe_prompt_strategy}_"
        f"steps{num_inference_steps}_cfg{format_generation_value(guidance_scale)}_"
        f"cnet{format_generation_value(controlnet_conditioning_scale)}_"
        f"seed{format_generation_value(seed)}{suffix}"
    )


def resolve_output_path(
    output_path: str | Path,
    sketch_id: str,
    model_key: str,
    prompt_strategy: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int | None,
) -> Path:
    """Return an output path whose filename records generation settings."""
    requested = Path(output_path)
    output_dir = requested if not requested.suffix else requested.parent
    return output_dir / build_generation_filename(
        sketch_id=sketch_id,
        model_key=model_key,
        prompt_strategy=prompt_strategy,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=seed,
    )


def load_controlnet_model(
    model_name: str = "lllyasviel/sd-controlnet-canny",
    dtype: str | torch.dtype | None = None,
    device: str | None = None,
) -> ControlNetModel:
    """Load a ControlNet model for sketch/line art guidance."""
    resolved_dtype = resolve_dtype(dtype, device)
    controlnet = ControlNetModel.from_pretrained(
        model_name,
        torch_dtype=resolved_dtype,
        use_auth_token=os.environ.get("HUGGINGFACE_TOKEN"),
    )
    return controlnet


def load_pipeline(
    model_key: str = "sd_v1_5",
    base_model: str | None = None,
    controlnet_model: str | None = None,
    device: str | None = None,
    dtype: str | torch.dtype | None = None,
) -> StableDiffusionControlNetPipeline | StableDiffusionXLControlNetPipeline:
    """
    Load Stable Diffusion + ControlNet pipeline.
    
    Uses fp16 on GPU, fp32 on CPU.
    """
    model_config = resolve_model_config(model_key, base_model, controlnet_model)
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    resolved_base_model = model_config["base_model"]
    resolved_controlnet_model = model_config["controlnet_model"]
    pipeline_type = model_config.get("pipeline", "sd")

    cache_key = (
        model_key,
        pipeline_type,
        resolved_base_model,
        resolved_controlnet_model,
        resolved_device,
        str(resolved_dtype),
    )
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    controlnet = load_controlnet_model(
        resolved_controlnet_model,
        dtype=resolved_dtype,
        device=resolved_device,
    )

    pipeline_cls = (
        StableDiffusionXLControlNetPipeline
        if pipeline_type == "sdxl"
        else StableDiffusionControlNetPipeline
    )
    pipe = pipeline_cls.from_pretrained(
        resolved_base_model,
        controlnet=controlnet,
        torch_dtype=resolved_dtype,
        use_auth_token=os.environ.get("HUGGINGFACE_TOKEN"),
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolved_device)
    
    # Enable memory optimizations
    if resolved_device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            # xformers not available, that's fine
            pass
    
    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def preprocess_sketch(sketch_path: str | Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """
    Preprocess CAD sketch:
    - Load image
    - Convert to grayscale
    - Resize to the model's expected conditioning size
    - Optional: Apply line detection for cleaner edges
    """
    sketch = Image.open(sketch_path).convert("L")  # Grayscale
    sketch = sketch.resize(size, Image.Resampling.LANCZOS)
    return sketch.convert("RGB")


def generate_fashion_design(
    sketch_path: str | Path,
    prompt: str,
    output_path: str | Path,
    model_key: str = "sd_v1_5",
    base_model: str | None = None,
    controlnet_model: str | None = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
    seed: int = None,
    prompt_strategy: str = "llm",
    sketch_id: str | None = None,
    device: str | None = None,
    dtype: str | torch.dtype | None = None,
) -> dict[str, Any]:
    """
    Generate a fashion design using Stable Diffusion + ControlNet.
    
    Args:
        sketch_path: Path to the CAD sketch image
        prompt: Gemini-generated detailed fashion prompt
        output_path: Where to save the generated image
        model_key: Simple key from MODEL_REGISTRY
        base_model: Optional Diffusers base model ID/path override
        controlnet_model: Optional Diffusers ControlNet model ID/path override
        num_inference_steps: Denoising steps (higher = better quality, slower)
        guidance_scale: How strongly to follow the prompt (7.5 is typical)
        controlnet_conditioning_scale: How much to follow the sketch (1.0 = full control)
        negative_prompt: What NOT to generate
        seed: For reproducibility
        prompt_strategy: Prompt strategy used to create the prompt
        sketch_id: Stable sketch ID for reproducible filenames
        device: Optional device override
        dtype: Optional torch dtype or dtype string
    
    Returns:
        Metadata dictionary containing generation parameters and output path
    """
    model_config = resolve_model_config(model_key, base_model, controlnet_model)
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    resolved_sketch_id = sketch_id or Path(sketch_path).stem
    image_size = int(model_config.get("image_size", 512))
    final_output_path = resolve_output_path(
        output_path=output_path,
        sketch_id=resolved_sketch_id,
        model_key=model_key,
        prompt_strategy=prompt_strategy,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=seed,
    )

    metadata: dict[str, Any] = {
        "sketch_id": resolved_sketch_id,
        "sketch_path": str(sketch_path),
        "prompt_strategy": prompt_strategy,
        "model_key": model_key,
        "base_model": model_config["base_model"],
        "controlnet_model": model_config["controlnet_model"],
        "pipeline": model_config.get("pipeline", "sd"),
        "image_size": image_size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "output_path": str(final_output_path),
        "device": resolved_device,
        "dtype": str(resolved_dtype),
    }
    logger.info("Generating fashion design with parameters: %s", metadata)

    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(seed)
    else:
        generator = None
    
    # Preprocess sketch
    sketch = preprocess_sketch(sketch_path, size=(image_size, image_size))
    
    # Load pipeline (or reuse from cache if available)
    pipe = load_pipeline(
        model_key=model_key,
        base_model=model_config["base_model"],
        controlnet_model=model_config["controlnet_model"],
        device=resolved_device,
        dtype=resolved_dtype,
    )
    
    # Generate
    with torch.no_grad() if resolved_device == "cuda" else torch.inference_mode():
        image = pipe(
            prompt=prompt,
            image=sketch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]
    
    # Save
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(final_output_path)
    
    return metadata


def generate_design_with_variations(
    sketch_path: str | Path,
    prompt: str,
    output_dir: str | Path,
    num_variations: int = 3,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    model_key: str = "sd_v1_5",
    base_model: str | None = None,
    controlnet_model: str | None = None,
    prompt_strategy: str = "llm",
) -> list[dict[str, Any]]:
    """
    Generate multiple design variations with different seeds.
    
    Returns list of generation metadata dictionaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    for i in range(num_variations):
        try:
            metadata = generate_fashion_design(
                sketch_path=sketch_path,
                prompt=prompt,
                output_path=output_dir,
                model_key=model_key,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=42 + i,  # Different seed per variation
                prompt_strategy=prompt_strategy,
                sketch_id=f"{Path(sketch_path).stem}_variation_{i+1}",
                base_model=base_model,
                controlnet_model=controlnet_model,
            )
            records.append(metadata)
        except Exception as e:
            print(f"Error generating variation {i+1}: {e}")
    
    return records


def _safe_filename_part(value: Any) -> str:
    text = str(value) if value not in (None, "") else "none"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
