"""CLI to generate fashion designs with image generation.

Usage: python tools/generate_design.py <sketch_image_path>
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.generator import build_prompt_for_strategy, normalize_prompt_strategy
from backend.app.tones import KANSEI_WORDS, TONES


MODEL_KEYS = ["sd_v1_5", "sdxl", "ssd_1b"]
DEFAULT_MODEL_KEY = "sd_v1_5"
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CONTROLNET_CONDITIONING_SCALE = 1.0
DEFAULT_SEED = 42


def pick_from_list(title, options, max_selections=None):
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")

    if max_selections:
        prompt_text = f"Select comma-separated numbers (max {max_selections}, or press Enter to skip): "
    else:
        prompt_text = "Select comma-separated numbers (or press Enter to skip): "

    raw = input(prompt_text).strip()
    if not raw:
        return []
    try:
        picks = [int(x.strip()) for x in raw.split(",") if x.strip()]
        selected = [options[i - 1] for i in picks if 1 <= i <= len(options)]

        if max_selections and len(selected) > max_selections:
            print(f"Too many selections! Maximum is {max_selections}. Try again.")
            return pick_from_list(title, options, max_selections)

        return selected
    except Exception:
        print("Invalid selection. Try again.")
        return pick_from_list(title, options, max_selections)


def pick_prompt_strategy():
    raw = input("\nPrompt strategy [llm/rule_based] (default: llm): ").strip() or "llm"
    try:
        return normalize_prompt_strategy(raw)
    except ValueError as e:
        print(e)
        return pick_prompt_strategy()


def main():
    args = parse_args()

    print("=" * 60)
    print("Fashion Design Generator - CLI Version")
    print("=" * 60)

    if args.sketch_image_path:
        sketch_path = Path(args.sketch_image_path)
    else:
        sketch_input = input("\nEnter path to your sketch image: ").strip()
        sketch_path = Path(sketch_input)

    if not sketch_path.exists():
        print(f"Error: Sketch file not found at {sketch_path}")
        sys.exit(1)

    print(f"\nUsing sketch: {sketch_path}")

    print("\n" + "=" * 60)
    tones = pick_from_list("Available Tones (select up to 3):", TONES, max_selections=3)
    kansei = pick_from_list("Available Kansei words (select any):", KANSEI_WORDS)

    if not tones and not kansei:
        print("\nError: Please select at least one tone or Kansei word.")
        sys.exit(1)

    prompt_strategy = pick_prompt_strategy()
    print(f"\nSelected tones: {', '.join(tones) if tones else 'None'}")
    print(f"Selected Kansei words: {', '.join(kansei) if kansei else 'None'}")
    print(f"Prompt strategy: {prompt_strategy}")

    print("\n" + "=" * 60)
    print(f"Generating prompt with {prompt_strategy} strategy...")
    print("=" * 60)

    try:
        image_prompt, prompt_strategy = build_prompt_for_strategy(
            tones,
            kansei,
            prompt_strategy=prompt_strategy,
            sketch_path=sketch_path,
        )

        print("\n--- Generated Prompt ---")
        print(image_prompt)
        print()
    except Exception as e:
        print(f"\nError generating prompt: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Generating image with Stable Diffusion + ControlNet...")
    print("=" * 60)

    try:
        from backend.app.image_generator import generate_fashion_design

        output_path = Path(PROJECT_ROOT) / "backend" / "media" / "generated"
        output_path.mkdir(parents=True, exist_ok=True)

        print("\nProcessing... This may take a few minutes...")
        print(f"Input: {sketch_path}")
        print(f"Output directory: {output_path}")

        metadata = generate_fashion_design(
            sketch_path=sketch_path,
            prompt=image_prompt,
            output_path=output_path,
            model_key=args.model_key,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            seed=args.seed,
            prompt_strategy=prompt_strategy,
            sketch_id=sketch_path.stem,
        )
        generated_path = Path(metadata["output_path"])

        print("\nSuccess! Generated image saved to:")
        print(f"  {generated_path}")

        prompt_path = generated_path.with_suffix(".txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(f"Prompt Strategy: {prompt_strategy}\n")
            f.write(f"Generation Metadata: {metadata}\n")
            f.write(f"Tones: {', '.join(tones)}\n")
            f.write(f"Kansei Words: {', '.join(kansei)}\n\n")
            f.write(f"Generated Prompt:\n{image_prompt}")
        print("Prompt saved to:")
        print(f"  {prompt_path}")

    except ImportError as e:
        print("\nError: Could not load image generation dependencies.")
        print(f"Details: {e}")
        print("\nThis is likely due to PyTorch not being properly installed.")
        print("The prompt has been generated but image generation is unavailable.")

        prompt_path = save_failed_prompt(image_prompt, sketch_path)
        print(f"\nPrompt saved to: {prompt_path}")
        print("\nYou can use this prompt with other image generation tools.")

    except Exception as e:
        print(f"\nError during image generation: {e}")
        print("\nThe prompt was generated successfully but image generation failed.")

        prompt_path = save_failed_prompt(image_prompt, sketch_path)
        print(f"\nPrompt saved to: {prompt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one fashion design from a CAD sketch.")
    parser.add_argument("sketch_image_path", nargs="?")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY, choices=MODEL_KEYS)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument(
        "--controlnet-conditioning-scale",
        type=float,
        default=DEFAULT_CONTROLNET_CONDITIONING_SCALE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def save_failed_prompt(image_prompt: str, sketch_path: Path) -> Path:
    prompt_dir = Path(PROJECT_ROOT) / "backend" / "media" / "generated"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"{sketch_path.stem}_failed_prompt.txt"
    prompt_path.write_text(image_prompt, encoding="utf-8")
    return prompt_path


if __name__ == "__main__":
    main()
