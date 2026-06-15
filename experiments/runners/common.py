"""Shared helpers for experiment runners."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

from experiments.logging.experiment_logger import ExperimentLogger
from experiments.metrics.clip_score import compute_clip_score
from experiments.metrics.lpips_distance import compute_lpips_distance
from experiments.metrics.runtime import measure_runtime
from experiments.metrics.sketch_iou import compute_sketch_iou
from experiments.prompts.gemini_prompt_builder import generate_gemini_image_prompt
from experiments.prompts.rule_based_templates import build_rule_based_prompt
from backend.app.generator import normalize_prompt_strategy


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = _resolve_path(path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def clone_config_with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        target = cloned
        parts = dotted_key.split("__")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return cloned


def load_manifest(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    manifest_path = _resolve_path(path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if not row.get("sketch_path"):
                continue
            rows.append(_case_from_row(row, index))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def run_case(
    config: dict[str, Any],
    case: dict[str, Any],
    logger: ExperimentLogger,
    prompt_only: bool = False,
) -> dict[str, Any]:
    model = config.get("model", {})
    generation = config.get("generation", {})
    prompt_config = config.get("prompt", {})
    metrics = config.get("metrics", [])
    seed = generation.get("seed")
    model_label = model.get("label", "model")
    case_id = case["sketch_id"]
    prompt_strategy = get_prompt_strategy(prompt_config)

    prompt = build_prompt(config, case)
    prompt_stem = f"{case_id}_{model_label}_seed{seed}_{prompt_strategy}"
    prompt_path = logger.write_prompt(prompt_stem, prompt)

    record: dict[str, Any] = {
        "status": "prompt_only" if prompt_only else "pending",
        "run_id": logger.run_id,
        "case": case,
        "model": model,
        "generation": generation,
        "prompt": {
            "prompt_strategy": prompt_strategy,
            "path": str(prompt_path),
            "text": prompt,
        },
        "metrics": {},
        "errors": {},
    }

    if prompt_only:
        logger.append_record(record)
        return record

    output_path = logger.images_dir / f"{case_id}_{model_label}_seed{seed}.png"

    try:
        from backend.app.image_generator import generate_fashion_design

        with measure_runtime() as runtime:
            generate_fashion_design(
                sketch_path=case["sketch_path"],
                prompt=prompt,
                output_path=output_path,
                num_inference_steps=generation.get("num_inference_steps", 20),
                guidance_scale=generation.get("guidance_scale", 7.5),
                controlnet_conditioning_scale=generation.get(
                    "controlnet_conditioning_scale", 1.0
                ),
                negative_prompt=generation.get(
                    "negative_prompt", "blurry, low quality, distorted, ugly, bad anatomy"
                ),
                seed=seed,
                base_model=model.get("base_model", "runwayml/stable-diffusion-v1-5"),
                controlnet_model=model.get("controlnet_model", "lllyasviel/sd-controlnet-canny"),
            )

        record["status"] = "ok"
        record["output_path"] = str(output_path)
        record["runtime"] = runtime
        record["metrics"] = compute_selected_metrics(metrics, case["sketch_path"], output_path, prompt)
    except Exception as exc:
        record["status"] = "error"
        record["errors"]["generation"] = str(exc)

    logger.append_record(record)
    return record


def build_prompt(config: dict[str, Any], case: dict[str, Any]) -> str:
    prompt_config = config.get("prompt", {})
    prompt_strategy = get_prompt_strategy(prompt_config)

    if prompt_strategy == "manual":
        prompt = case.get("manual_prompt")
        if not prompt:
            raise ValueError(f"Case {case['sketch_id']} has no manual_prompt.")
        return prompt

    if prompt_strategy == "rule_based":
        return build_rule_based_prompt(
            tones=case.get("tones", []),
            kansei_words=case.get("kansei_words", []),
            garment_type=case.get("garment_type") or "fashion garment",
            sketch_path=case["sketch_path"],
        )

    if prompt_strategy == "llm":
        return generate_gemini_image_prompt(
            tones=case.get("tones", []),
            kansei_words=case.get("kansei_words", []),
            sketch_path=case["sketch_path"],
            model=prompt_config.get("gemini_model", "gemini-2.5-flash"),
        )

    raise ValueError(f"Unknown prompt_strategy: {prompt_strategy}")


def get_prompt_strategy(prompt_config: dict[str, Any]) -> str:
    """Return normalized strategy, accepting old experiment config aliases."""
    strategy = prompt_config.get("prompt_strategy", prompt_config.get("mode", "llm"))
    if strategy == "manual":
        return "manual"
    return normalize_prompt_strategy(strategy)


def compute_selected_metrics(
    metric_names: list[str],
    sketch_path: str | Path,
    generated_path: str | Path,
    prompt: str,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for metric_name in metric_names:
        try:
            if metric_name == "sketch_iou":
                values[metric_name] = compute_sketch_iou(sketch_path, generated_path)
            elif metric_name == "clip":
                values[metric_name] = compute_clip_score(prompt, generated_path)
            elif metric_name == "lpips":
                values[metric_name] = compute_lpips_distance(sketch_path, generated_path)
            else:
                values[f"{metric_name}_error"] = f"Unknown metric: {metric_name}"
        except Exception as exc:
            values[f"{metric_name}_error"] = str(exc)
    return values


def make_logger(config: dict[str, Any], suffix: str | None = None) -> ExperimentLogger:
    name = config.get("experiment_name", "experiment")
    if suffix:
        name = f"{name}_{suffix}"
    logger = ExperimentLogger(name)
    logger.write_json("config.json", config)
    return logger


def _case_from_row(row: dict[str, str], index: int) -> dict[str, Any]:
    sketch_id = row.get("sketch_id") or f"sketch_{index:03d}"
    sketch_path = _resolve_path(row["sketch_path"])
    return {
        "sketch_id": sketch_id,
        "sketch_path": str(sketch_path),
        "garment_type": row.get("garment_type", ""),
        "tones": _split_list(row.get("tones", "")),
        "kansei_words": _split_list(row.get("kansei_words", "")),
        "manual_prompt": row.get("manual_prompt", ""),
    }


def _split_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item.strip() for item in value if item.strip()]
    normalized = value.replace(",", ";")
    return [item.strip() for item in normalized.split(";") if item.strip()]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
