"""Summarize experiment raw results into CSV and paper-ready markdown tables.

Default input:
    experiments/results/raw_results.csv

Outputs:
    experiments/results/summary_by_model.csv
    experiments/results/summary_by_prompt_strategy.csv
    experiments/results/summary_by_model_and_prompt.csv
    experiments/results/summary_by_controlnet_scale.csv
    experiments/results/summary_by_steps.csv
    experiments/results/paper_tables.md
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "experiments" / "results" / "raw_results.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

METRIC_COLUMNS = {
    "clip_score": ("mean_clip_score", "std_clip_score"),
    "lpips_score": ("mean_lpips_score", "std_lpips_score"),
    "sketch_iou": ("mean_sketch_iou", "std_sketch_iou"),
    "latency_seconds": ("mean_latency_seconds", "std_latency_seconds"),
    "peak_vram_gb": ("mean_peak_vram_gb", None),
}

SUMMARY_SPECS = [
    ("summary_by_model.csv", ("model_key",)),
    ("summary_by_prompt_strategy.csv", ("prompt_strategy",)),
    ("summary_by_model_and_prompt.csv", ("model_key", "prompt_strategy")),
    ("summary_by_controlnet_scale.csv", ("controlnet_conditioning_scale",)),
    ("summary_by_steps.csv", ("num_inference_steps",)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment raw_results.csv.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_path)
    summaries: dict[str, list[dict[str, Any]]] = {}
    for filename, group_keys in SUMMARY_SPECS:
        summary_rows = summarize(rows, group_keys)
        summaries[filename] = summary_rows
        write_summary_csv(output_dir / filename, group_keys, summary_rows)

    write_paper_tables(output_dir / "paper_tables.md", summaries)
    print(f"Summaries written to {output_dir}")


def read_rows(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Raw results CSV not found: {input_path}")
    with input_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not is_successful_row(row):
            continue
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key_values, group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            group_key: key_values[index] for index, group_key in enumerate(group_keys)
        }
        for source_column, (mean_column, std_column) in METRIC_COLUMNS.items():
            values = numeric_values(group_rows, source_column)
            summary[mean_column] = safe_mean(values)
            if std_column:
                summary[std_column] = safe_std(values)

        summary["number_of_runs"] = len(group_rows)
        summary_rows.append(summary)

    return summary_rows


def is_successful_row(row: dict[str, str]) -> bool:
    """Summaries should describe completed generations, not failed attempts."""
    status = row.get("status")
    return status == "ok" or (status in (None, "") and bool(row.get("generated_image_path")))


def write_summary_csv(
    output_path: Path,
    group_keys: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = list(group_keys) + [
        "mean_clip_score",
        "std_clip_score",
        "mean_lpips_score",
        "std_lpips_score",
        "mean_sketch_iou",
        "std_sketch_iou",
        "mean_latency_seconds",
        "std_latency_seconds",
        "mean_peak_vram_gb",
        "number_of_runs",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field)) for field in fieldnames})


def write_paper_tables(output_path: Path, summaries: dict[str, list[dict[str, Any]]]) -> None:
    sections = [
        (
            "Table A: Prompt Strategy Comparison",
            summaries["summary_by_prompt_strategy.csv"],
            ("prompt_strategy",),
        ),
        (
            "Table B: Model Comparison",
            summaries["summary_by_model.csv"],
            ("model_key",),
        ),
        (
            "Table C: ControlNet Scale Ablation",
            summaries["summary_by_controlnet_scale.csv"],
            ("controlnet_conditioning_scale",),
        ),
        (
            "Table D: Inference Step Ablation",
            summaries["summary_by_steps.csv"],
            ("num_inference_steps",),
        ),
    ]

    lines = [
        "# Paper Tables",
        "",
        "Interpretation notes:",
        "- Higher CLIP is better.",
        "- Lower LPIPS is better.",
        "- Higher Sketch IoU is better.",
        "- Lower latency and VRAM are better for deployment.",
        "",
    ]

    for title, rows, group_keys in sections:
        lines.append(f"## {title}")
        lines.extend(markdown_table(rows, group_keys))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[str]:
    columns = list(group_keys) + [
        "mean_clip_score",
        "std_clip_score",
        "mean_lpips_score",
        "std_lpips_score",
        "mean_sketch_iou",
        "std_sketch_iou",
        "mean_latency_seconds",
        "std_latency_seconds",
        "mean_peak_vram_gb",
        "number_of_runs",
    ]
    headers = [human_label(column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    if not rows:
        lines.append("| " + " | ".join("" for _ in columns) + " |")
    return lines


def numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_std(values: list[float]) -> float | None:
    return stdev(values) if len(values) > 1 else None


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def human_label(column: str) -> str:
    labels = {
        "model_key": "Model",
        "prompt_strategy": "Prompt Strategy",
        "controlnet_conditioning_scale": "ControlNet Scale",
        "num_inference_steps": "Steps",
        "mean_clip_score": "Mean CLIP",
        "std_clip_score": "Std CLIP",
        "mean_lpips_score": "Mean LPIPS",
        "std_lpips_score": "Std LPIPS",
        "mean_sketch_iou": "Mean Sketch IoU",
        "std_sketch_iou": "Std Sketch IoU",
        "mean_latency_seconds": "Mean Latency (s)",
        "std_latency_seconds": "Std Latency (s)",
        "mean_peak_vram_gb": "Mean Peak VRAM (GB)",
        "number_of_runs": "Runs",
    }
    return labels.get(column, column)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


if __name__ == "__main__":
    main()
