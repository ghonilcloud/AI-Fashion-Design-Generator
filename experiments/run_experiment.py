"""Run the full paper experiment grid over a folder of CAD sketches.

Examples:
    python experiments/run_experiment.py --dataset datasets/cad_sketches --limit 5
    python experiments/run_experiment.py --dataset datasets/cad_sketches --overwrite
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.generator import build_prompt_for_strategy
from experiments.metrics import evaluate_generation
from experiments.metrics.runtime import measure_runtime


PROMPT_STRATEGIES = ["rule_based", "llm"]
MODEL_KEYS = ["sd_v1_5", "sdxl", "ssd_1b"]
NUM_INFERENCE_STEPS = [10, 20]
GUIDANCE_SCALES = [7.5]
CONTROLNET_CONDITIONING_SCALES = [0.5, 0.75, 1.0]
SEEDS = [42]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_REGISTRY_FALLBACK = {
    "sd_v1_5": {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "controlnet_model": "lllyasviel/sd-controlnet-canny",
    },
    "sdxl": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
    },
    "ssd_1b": {
        "base_model": "segmind/SSD-1B",
        "controlnet_model": "diffusers/controlnet-canny-sdxl-1.0",
    },
}

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RAW_RESULTS_CSV = RESULTS_DIR / "raw_results.csv"
OUTPUT_IMAGES_DIR = RESULTS_DIR / "images"
PROMPTS_DIR = RESULTS_DIR / "prompts"
METADATA_DIR = RESULTS_DIR / "metadata"

CSV_FIELDS = [
    "run_id",
    "sketch_id",
    "sketch_path",
    "prompt_strategy",
    "model_key",
    "base_model",
    "controlnet_model",
    "num_inference_steps",
    "guidance_scale",
    "controlnet_conditioning_scale",
    "seed",
    "prompt",
    "generated_image_path",
    "clip_score",
    "lpips_score",
    "sketch_iou",
    "latency_seconds",
    "peak_vram_gb",
    "timestamp",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full CAD-sketch fashion experiment grid.")
    parser.add_argument("--dataset", default="datasets/cad_sketches")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tones", default="")
    parser.add_argument("--kansei-words", default="")
    args = parser.parse_args()

    setup_output_dirs()
    sketches = find_sketches(args.dataset, limit=args.limit)
    existing_run_ids = load_existing_run_ids(RAW_RESULTS_CSV)
    tones = split_cli_list(args.tones)
    kansei_words = split_cli_list(args.kansei_words)

    print(f"Found {len(sketches)} sketches.")
    print(f"Writing raw results to {RAW_RESULTS_CSV}")

    completed = 0
    skipped = 0
    failed = 0

    for sketch_path in sketches:
        sketch_id = safe_id(sketch_path.stem)
        for combo in iter_grid():
            run_id = build_run_id(sketch_id=sketch_id, **combo)
            if run_id in existing_run_ids and not args.overwrite:
                skipped += 1
                print(f"Skipping existing run: {run_id}")
                continue

            if args.overwrite and run_id in existing_run_ids:
                remove_run_id_from_csv(RAW_RESULTS_CSV, run_id)
                existing_run_ids.discard(run_id)

            try:
                row = run_one(
                    run_id=run_id,
                    sketch_id=sketch_id,
                    sketch_path=sketch_path,
                    tones=tones,
                    kansei_words=kansei_words,
                    **combo,
                )
                append_csv_row(RAW_RESULTS_CSV, row)
                existing_run_ids.add(run_id)
                completed += 1
                print(f"Completed: {run_id}")
            except Exception as exc:
                failed += 1
                error_row = build_error_row(run_id, sketch_id, sketch_path, combo, exc)
                append_csv_row(RAW_RESULTS_CSV, error_row)
                existing_run_ids.add(run_id)
                print(f"Failed: {run_id}: {exc}")

    print(f"Done. completed={completed}, skipped={skipped}, failed={failed}")


def run_one(
    run_id: str,
    sketch_id: str,
    sketch_path: Path,
    prompt_strategy: str,
    model_key: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int,
    tones: list[str],
    kansei_words: list[str],
) -> dict[str, Any]:
    from backend.app.image_generator import generate_fashion_design

    model_config = resolve_model_config_for_row(model_key)
    timestamp = datetime.now().isoformat(timespec="seconds")

    prompt, normalized_prompt_strategy = build_prompt_for_strategy(
        selected_tones=tones,
        selected_kansei=kansei_words,
        prompt_strategy=prompt_strategy,
        sketch_path=sketch_path,
        sketch_metadata=f"Dataset sketch file: {sketch_path.name}",
    )

    prompt_path = PROMPTS_DIR / f"{run_id}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    with measure_runtime() as runtime:
        generation_metadata = generate_fashion_design(
            sketch_path=sketch_path,
            prompt=prompt,
            output_path=OUTPUT_IMAGES_DIR,
            model_key=model_key,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed,
            prompt_strategy=normalized_prompt_strategy,
            sketch_id=sketch_id,
        )

    generated_image_path = Path(generation_metadata["output_path"])
    metrics = evaluate_generation(
        sketch_path=sketch_path,
        generated_image_path=generated_image_path,
        prompt=prompt,
        metrics_config={
            "compute_clip": True,
            "compute_lpips": True,
            "compute_sketch_iou": True,
            "latency_seconds": runtime.get("elapsed_seconds"),
            "peak_vram_gb": runtime.get("cuda_peak_memory_gb"),
        },
    )

    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "sketch_id": sketch_id,
        "sketch_path": str(sketch_path),
        "prompt_path": str(prompt_path),
        "generation_metadata": generation_metadata,
        "metrics": metrics,
    }
    metadata_path = METADATA_DIR / f"{run_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "run_id": run_id,
        "sketch_id": sketch_id,
        "sketch_path": str(sketch_path),
        "prompt_strategy": normalized_prompt_strategy,
        "model_key": model_key,
        "base_model": model_config["base_model"],
        "controlnet_model": model_config["controlnet_model"],
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "seed": seed,
        "prompt": prompt,
        "generated_image_path": str(generated_image_path),
        "clip_score": metrics.get("clip_score"),
        "lpips_score": metrics.get("lpips_score"),
        "sketch_iou": metrics.get("sketch_iou"),
        "latency_seconds": metrics.get("latency_seconds"),
        "peak_vram_gb": metrics.get("peak_vram_gb"),
        "timestamp": timestamp,
    }


def iter_grid():
    for (
        prompt_strategy,
        model_key,
        num_inference_steps,
        guidance_scale,
        controlnet_conditioning_scale,
        seed,
    ) in product(
        PROMPT_STRATEGIES,
        MODEL_KEYS,
        NUM_INFERENCE_STEPS,
        GUIDANCE_SCALES,
        CONTROLNET_CONDITIONING_SCALES,
        SEEDS,
    ):
        yield {
            "prompt_strategy": prompt_strategy,
            "model_key": model_key,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "seed": seed,
        }


def find_sketches(dataset: str | Path, limit: int | None = None) -> list[Path]:
    dataset_path = resolve_path(dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a folder: {dataset_path}")

    sketches = sorted(
        path for path in dataset_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        sketches = sketches[:limit]
    return sketches


def build_run_id(
    sketch_id: str,
    prompt_strategy: str,
    model_key: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int,
) -> str:
    return (
        f"{sketch_id}__{prompt_strategy}__{model_key}__"
        f"steps{num_inference_steps}__cfg{format_value(guidance_scale)}__"
        f"cnet{format_value(controlnet_conditioning_scale)}__seed{seed}"
    )


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def load_existing_run_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {row["run_id"] for row in csv.DictReader(handle) if row.get("run_id")}


def remove_run_id_from_csv(csv_path: Path, run_id: str) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("run_id") != run_id]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_error_row(
    run_id: str,
    sketch_id: str,
    sketch_path: Path,
    combo: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    model_config = resolve_model_config_for_row(combo["model_key"])
    timestamp = datetime.now().isoformat(timespec="seconds")
    metadata_path = METADATA_DIR / f"{run_id}.json"
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "sketch_id": sketch_id,
        "sketch_path": str(sketch_path),
        "parameters": combo,
        "error": str(exc),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "run_id": run_id,
        "sketch_id": sketch_id,
        "sketch_path": str(sketch_path),
        "prompt_strategy": combo["prompt_strategy"],
        "model_key": combo["model_key"],
        "base_model": model_config["base_model"],
        "controlnet_model": model_config["controlnet_model"],
        "num_inference_steps": combo["num_inference_steps"],
        "guidance_scale": combo["guidance_scale"],
        "controlnet_conditioning_scale": combo["controlnet_conditioning_scale"],
        "seed": combo["seed"],
        "prompt": f"ERROR: {exc}",
        "timestamp": timestamp,
    }


def setup_output_dirs() -> None:
    for path in (RESULTS_DIR, OUTPUT_IMAGES_DIR, PROMPTS_DIR, METADATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_model_config_for_row(model_key: str) -> dict[str, str]:
    try:
        from backend.app.image_generator import resolve_model_config

        return resolve_model_config(model_key)
    except Exception:
        if model_key not in MODEL_REGISTRY_FALLBACK:
            allowed = ", ".join(sorted(MODEL_REGISTRY_FALLBACK))
            raise ValueError(f"Unknown model_key '{model_key}'. Use one of: {allowed}.")
        model_config = MODEL_REGISTRY_FALLBACK[model_key].copy()
        model_config["model_key"] = model_key
        return model_config


def split_cli_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def format_value(value: float | int) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
