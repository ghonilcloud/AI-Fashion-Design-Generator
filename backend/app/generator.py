"""Build prompts using preset tones/Kansei words and optional Gemini expansion."""
from typing import List
import base64
from pathlib import Path

from .tones import TONES, KANSEI_WORDS
from .llm_client import get_genai_client


VALID_PROMPT_STRATEGIES = {"llm", "rule_based"}


def normalize_prompt_strategy(prompt_strategy: str | None = "llm") -> str:
    """Normalize supported prompt-strategy names and aliases."""
    strategy = (prompt_strategy or "llm").strip().lower()
    aliases = {
        "gemini": "llm",
        "llm_prompt": "llm",
        "rule": "rule_based",
        "rule_based_prompt": "rule_based",
    }
    strategy = aliases.get(strategy, strategy)
    if strategy not in VALID_PROMPT_STRATEGIES:
        allowed = ", ".join(sorted(VALID_PROMPT_STRATEGIES))
        raise ValueError(f"Unsupported prompt_strategy '{prompt_strategy}'. Use one of: {allowed}.")
    return strategy


def build_gemini_instruction(
    selected_tones: List[str], 
    selected_kansei: List[str],
    sketch_path: str | Path = None,
) -> str:
    """Construct an instruction for Gemini to expand into a detailed image prompt.

    The instruction asks Gemini to produce a concise but highly-detailed image prompt
    suited for fashion sketches with realistic rendering while preserving the uploaded sketch silhouette.
    
    If sketch_path is provided, include it so Gemini can analyze the actual sketch shape.
    """
    tone_clause = ", ".join(selected_tones) if selected_tones else ""
    kansei_clause = ", ".join(selected_kansei) if selected_kansei else ""

    instructions = (
        f"You are a prompt engineer for fashion design generation. "
        f"Your task is to PRESERVE the silhouette and shape of the provided sketch while enhancing it with vibrant styling.\n\n"
        f"Selected tones/style: {tone_clause}\n"
        f"Selected Kansei (sensory/feeling) keywords: {kansei_clause}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. ANALYZE THE SKETCH SHAPE: Examine the uploaded sketch carefully. Identify the garment type, silhouette, sleeve type, hemline, and overall proportions.\n"
        "2. PRESERVE THE EXACT SILHOUETTE: The generated output MUST maintain the same silhouette, neckline, sleeve shape, and hemline as the uploaded sketch.\n"
        "3. RENDER WITH STYLE: Create a vibrant, well-rendered fashion design that applies the selected tones and Kansei through:\n"
        "   - Rich, appropriate color palettes that match the tones (e.g., 'Elegant' = sophisticated colors, 'Vibrant' = bold colors)\n"
        "   - Fabric textures and finishes (silk, cotton, structured, fluid, etc.)\n"
        "   - Realistic material appearance while keeping artistic clarity\n"
        "4. Output format: Produce a single, compact image-generation prompt (3-4 sentences) that describes:\n"
        "   - The EXACT silhouette, neckline, sleeves, and fit of the sketch\n"
        "   - The garment type and overall impression\n"
        "   - A rich color palette and fabric suggestions aligned with the tones/Kansei\n"
        "   - Details like draping, seams, textures, embellishments that enhance the style\n"
        "5. Visual style: Fashion illustration with realistic rendering, rich colors, artistic but wearable aesthetic.\n"
    )

    return instructions


def build_rule_based_prompt(
    selected_tones: List[str],
    selected_kansei: List[str],
    garment_type: str | None = None,
    sketch_metadata: str | None = None,
) -> str:
    """Build a deterministic non-LLM prompt for controlled experiments."""
    tones = ", ".join(selected_tones) if selected_tones else "none specified"
    kansei_words = ", ".join(selected_kansei) if selected_kansei else "none specified"
    garment_clause = f" The garment type is {garment_type.strip()}." if garment_type else ""
    metadata_clause = (
        f" Additional sketch metadata: {sketch_metadata.strip()}." if sketch_metadata else ""
    )

    return (
        "A CAD-style fashion technical drawing of a garment based on the uploaded sketch."
        f"{garment_clause}{metadata_clause} "
        "The design should preserve the original silhouette, neckline, sleeves, hemline, "
        "and construction lines. "
        f"The design should express the following tone descriptors: {tones}. "
        f"The design should express the following Kansei descriptors: {kansei_words}. "
        "Render as a clean fashion illustration with visible fabric texture, controlled colors, "
        "and a plain background."
    )


def call_gemini_for_prompt(instruction: str, sketch_path: str | Path = None, model: str = "gemini-2.5-flash") -> str:
    """Call Gemini API with optional image attachment."""
    client = get_genai_client()
    
    # If sketch provided, send it with the instruction so Gemini can analyze it
    if sketch_path and Path(sketch_path).exists():
        with open(sketch_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        # Determine mime type
        suffix = Path(sketch_path).suffix.lower()
        mime_type = "image/png" if suffix == ".png" else "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
        
        # Send with image
        response = client.models.generate_content(
            model=model,
            contents=[
                {"text": instruction},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data,
                    }
                },
            ],
        )
    else:
        response = client.models.generate_content(model=model, contents=instruction)
    
    return getattr(response, 'text', str(response))


def refine_for_image_model(gemini_text: str) -> str:
    """Optionally transform the Gemini output into a final image prompt for the image model.

    Appends rendering and quality instructions to produce vibrant, high-quality fashion designs.
    """
    suffix = (
        "\n--rendering: fashion illustration with realistic fabric textures and rich colors; "
        "artistic but wearable aesthetic; professional fashion design quality; "
        "maintain sketch silhouette exactly; vibrant and well-lit; focus on garment appeal and style."
    )
    return gemini_text.strip() + suffix


def build_prompt_for_strategy(
    selected_tones: List[str],
    selected_kansei: List[str],
    prompt_strategy: str = "llm",
    sketch_path: str | Path = None,
    garment_type: str | None = None,
    sketch_metadata: str | None = None,
    model: str = "gemini-2.5-flash",
) -> tuple[str, str]:
    """Build the final image prompt and return `(prompt, normalized_strategy)`."""
    strategy = normalize_prompt_strategy(prompt_strategy)
    if strategy == "rule_based":
        return (
            build_rule_based_prompt(
                selected_tones,
                selected_kansei,
                garment_type=garment_type,
                sketch_metadata=sketch_metadata,
            ),
            strategy,
        )

    instruction = build_gemini_instruction(selected_tones, selected_kansei, sketch_path=sketch_path)
    gemini_prompt = call_gemini_for_prompt(instruction, sketch_path=sketch_path, model=model)
    return refine_for_image_model(gemini_prompt), strategy
