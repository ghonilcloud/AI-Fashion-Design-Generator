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
from pathlib import Path
from PIL import Image
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import LineartDetector


_PIPELINE_CACHE = {}


def load_controlnet_model(model_name: str = "lllyasviel/sd-controlnet-canny") -> ControlNetModel:
    """Load a ControlNet model for sketch/line art guidance."""
    controlnet = ControlNetModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=os.environ.get("HUGGINGFACE_TOKEN"),
    )
    return controlnet


def load_pipeline(
    base_model: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model: str = "lllyasviel/sd-controlnet-canny",
) -> StableDiffusionControlNetPipeline:
    """
    Load Stable Diffusion + ControlNet pipeline.
    
    Uses fp16 on GPU, fp32 on CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    cache_key = (base_model, controlnet_model, device, str(dtype))
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    controlnet = load_controlnet_model(controlnet_model)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_auth_token=os.environ.get("HUGGINGFACE_TOKEN"),
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            # xformers not available, that's fine
            pass
    
    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def preprocess_sketch(sketch_path: str | Path) -> Image.Image:
    """
    Preprocess CAD sketch:
    - Load image
    - Convert to grayscale
    - Resize to 512x512 (standard SD size)
    - Optional: Apply line detection for cleaner edges
    """
    sketch = Image.open(sketch_path).convert("L")  # Grayscale
    sketch = sketch.resize((512, 512), Image.Resampling.LANCZOS)
    return sketch


def generate_fashion_design(
    sketch_path: str | Path,
    prompt: str,
    output_path: str | Path,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
    seed: int = None,
    base_model: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model: str = "lllyasviel/sd-controlnet-canny",
) -> str:
    """
    Generate a fashion design using Stable Diffusion + ControlNet.
    
    Args:
        sketch_path: Path to the CAD sketch image
        prompt: Gemini-generated detailed fashion prompt
        output_path: Where to save the generated image
        num_inference_steps: Denoising steps (higher = better quality, slower)
        guidance_scale: How strongly to follow the prompt (7.5 is typical)
        controlnet_conditioning_scale: How much to follow the sketch (1.0 = full control)
        negative_prompt: What NOT to generate
        seed: For reproducibility
        base_model: Diffusers base model ID/path
        controlnet_model: Diffusers ControlNet model ID/path
    
    Returns:
        Path to the generated image
    """
    
    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(seed)
    else:
        generator = None
    
    # Preprocess sketch
    sketch = preprocess_sketch(sketch_path)
    
    # Load pipeline (or reuse from cache if available)
    pipe = load_pipeline(base_model=base_model, controlnet_model=controlnet_model)
    
    # Generate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad() if device == "cuda" else torch.inference_mode():
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    return str(output_path)


def generate_design_with_variations(
    sketch_path: str | Path,
    prompt: str,
    output_dir: str | Path,
    num_variations: int = 3,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    base_model: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model: str = "lllyasviel/sd-controlnet-canny",
) -> list[str]:
    """
    Generate multiple design variations with different seeds.
    
    Returns list of paths to generated images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i in range(num_variations):
        output_path = output_dir / f"variation_{i+1}.png"
        try:
            path = generate_fashion_design(
                sketch_path=sketch_path,
                prompt=prompt,
                output_path=output_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=42 + i,  # Different seed per variation
                base_model=base_model,
                controlnet_model=controlnet_model,
            )
            paths.append(path)
        except Exception as e:
            print(f"Error generating variation {i+1}: {e}")
    
    return paths
