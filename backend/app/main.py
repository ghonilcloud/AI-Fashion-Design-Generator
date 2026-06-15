from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
from typing import List

from .config import UPLOADS_DIR, GENERATED_DIR, MEDIA_DIR
from .models import GenerateDesignResponse
from .generator import build_prompt_for_strategy, normalize_prompt_strategy
from .tones import TONES, KANSEI_WORDS

# Optional import - only load when needed to avoid PyTorch DLL issues
try:
    from .image_generator import generate_fashion_design, resolve_output_path
    IMAGE_GENERATION_AVAILABLE = True
except Exception as e:
    print(f"Warning: Image generation disabled. Error loading dependencies: {e}")
    IMAGE_GENERATION_AVAILABLE = False
    generate_fashion_design = None
    resolve_output_path = None

app = FastAPI(
    title="Fashion Emotion Design API",
    description="Backend for emotion-aware CAD-style fashion design generation.",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve media files
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


@app.post("/api/generate-design", response_model=GenerateDesignResponse)
async def generate_design(
    image: UploadFile = File(...),
    kansei_text: str = Form(...),
    style_profile: str | None = Form(None),
    tones: List[str] = Form(default=[]),
    kansei_words: List[str] = Form(default=[]),
    prompt_strategy: str = Form("llm"),
    garment_type: str | None = Form(None),
    model_key: str = Form("sd_v1_5"),
    base_model: str | None = Form(None),
    controlnet_model: str | None = Form(None),
    num_inference_steps: int = Form(20),
    guidance_scale: float = Form(7.5),
    controlnet_conditioning_scale: float = Form(1.0),
    seed: int | None = Form(None),
    negative_prompt: str = Form("blurry, low quality, distorted, ugly, bad anatomy"),
):
    """
    Main endpoint: receives sketch + kansei text + optional tones/Kansei words.
    
    - Saves the uploaded sketch
    - Builds a prompt using either Gemini or the deterministic rule-based template
    - Returns generated_image_url, the prompt, and the prompt strategy
    """

    # 1. Save the uploaded sketch
    sketch_id = uuid.uuid4().hex
    original_ext = Path(image.filename).suffix or ".png"
    sketch_filename = f"sketch_{sketch_id}{original_ext}"
    sketch_path = UPLOADS_DIR / sketch_filename

    with sketch_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 2. Build prompt using the selected strategy
    try:
        normalized_prompt_strategy = normalize_prompt_strategy(prompt_strategy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        llm_prompt, normalized_prompt_strategy = build_prompt_for_strategy(
            tones,
            kansei_words,
            prompt_strategy=normalized_prompt_strategy,
            sketch_path=sketch_path,
            garment_type=garment_type,
            sketch_metadata=style_profile,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prompt generation failed: {e}")

    # 3. Generate image using Stable Diffusion + ControlNet with sketch + prompt
    response_status = "ok"
    generation_metadata = None
    generation_error = None
    generated_path = _fallback_output_path(
        sketch_id=sketch_id,
        model_key=model_key,
        prompt_strategy=normalized_prompt_strategy,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=seed,
    )
    
    if IMAGE_GENERATION_AVAILABLE:
        try:
            generation_metadata = generate_fashion_design(
                sketch_path=sketch_path,
                prompt=llm_prompt,
                output_path=GENERATED_DIR,
                model_key=model_key,
                base_model=base_model,
                controlnet_model=controlnet_model,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                seed=seed,
                negative_prompt=negative_prompt,
                prompt_strategy=normalized_prompt_strategy,
                sketch_id=sketch_id,
            )
            generated_path = Path(generation_metadata["output_path"])
        except Exception as e:
            generation_error = str(e)
            response_status = "error"
            print(f"Image generation failed: {generation_error}")
            shutil.copyfile(sketch_path, generated_path)
    else:
        generation_error = "Image generation dependencies are unavailable."
        response_status = "error"
        print("Image generation unavailable - returning original sketch")
        shutil.copyfile(sketch_path, generated_path)

    # 4. Build URL for frontend
    generated_filename = generated_path.name
    generated_image_url = f"/media/generated/{generated_filename}"

    if generation_metadata is None:
        generation_metadata = _build_fallback_generation_metadata(
            sketch_id=sketch_id,
            sketch_path=sketch_path,
            generated_path=generated_path,
            prompt_strategy=normalized_prompt_strategy,
            model_key=model_key,
            base_model=base_model,
            controlnet_model=controlnet_model,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed,
            negative_prompt=negative_prompt,
            error=generation_error,
        )

    selected_model = generation_metadata.get("model_key", model_key)
    selected_base_model = generation_metadata.get("base_model", base_model)
    model_label = selected_base_model or selected_model
    if response_status == "ok":
        notes = (
            f"Generated using {selected_model} ({model_label}) + ControlNet with the "
            f"{normalized_prompt_strategy} prompt conditioned on your CAD sketch."
        )
    else:
        notes = (
            "Image generation failed or was unavailable; the returned image is the uploaded sketch copy. "
            f"Error: {generation_error}"
        )

    return GenerateDesignResponse(
        status=response_status,
        generated_image_url=generated_image_url,
        llm_prompt=llm_prompt,
        prompt_strategy=normalized_prompt_strategy,
        generation_metadata=generation_metadata,
        notes=notes,
    )


@app.get("/api/tones")
async def get_tones():
    """Return available tones and Kansei words for frontend dropdown."""
    return {"tones": TONES, "kansei_words": KANSEI_WORDS}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Mount frontend static files LAST so API routes take precedence
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    # Mount frontend at root so index.html is available at '/index.html' and '/'
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    # If the frontend folder is missing, leave API-only behavior and log a message
    print(f"Frontend directory not found at {FRONTEND_DIR}; serving API only.")


def _fallback_output_path(
    sketch_id: str,
    model_key: str,
    prompt_strategy: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int | None,
) -> Path:
    if resolve_output_path:
        return resolve_output_path(
            output_path=GENERATED_DIR,
            sketch_id=sketch_id,
            model_key=model_key,
            prompt_strategy=prompt_strategy,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed,
        )
    filename = (
        f"{sketch_id}_{model_key}_{prompt_strategy}_steps{num_inference_steps}_"
        f"cfg{str(guidance_scale).replace('.', 'p')}_"
        f"cnet{str(controlnet_conditioning_scale).replace('.', 'p')}_seed{seed}.png"
    )
    return GENERATED_DIR / filename


def _build_fallback_generation_metadata(
    sketch_id: str,
    sketch_path: Path,
    generated_path: Path,
    prompt_strategy: str,
    model_key: str,
    base_model: str | None,
    controlnet_model: str | None,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int | None,
    negative_prompt: str,
    error: str | None,
) -> dict:
    return {
        "status": "error",
        "sketch_id": sketch_id,
        "sketch_path": str(sketch_path),
        "prompt_strategy": prompt_strategy,
        "model_key": model_key,
        "base_model": base_model,
        "controlnet_model": controlnet_model,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "output_path": str(generated_path),
        "fallback_image_copied": True,
        "error": error,
    }
