"""CSV/JSONL logging helpers for experiment runners."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .system_info import collect_system_info


class ExperimentLogger:
    """Write reproducible per-run artifacts under `experiments/results`."""

    def __init__(self, experiment_name: str, results_root: str | Path = "experiments/results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment_name)
        self.run_id = f"{timestamp}_{safe_name}"
        self.results_root = Path(results_root)
        self.raw_dir = self.results_root / "raw" / self.run_id
        self.images_dir = self.results_root / "images" / self.run_id
        self.prompts_dir = self.results_root / "prompts" / self.run_id
        self.metrics_dir = self.results_root / "metrics" / self.run_id
        self.summary_dir = self.results_root / "summaries" / self.run_id

        for directory in (
            self.raw_dir,
            self.images_dir,
            self.prompts_dir,
            self.metrics_dir,
            self.summary_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.raw_dir / "records.jsonl"
        self.csv_path = self.raw_dir / "records.csv"
        self._csv_fields: list[str] | None = None

        self.write_json("system_info.json", collect_system_info(), directory=self.summary_dir)

    def write_json(
        self,
        filename: str,
        data: dict[str, Any],
        directory: Path | None = None,
    ) -> Path:
        path = (directory or self.raw_dir) / filename
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_prompt(self, stem: str, prompt: str) -> Path:
        path = self.prompts_dir / f"{stem}.txt"
        path.write_text(prompt, encoding="utf-8")
        return path

    def append_record(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        flat = _flatten_record(record)
        self._append_csv(flat)

    def _append_csv(self, flat_record: dict[str, Any]) -> None:
        if self._csv_fields is None:
            self._csv_fields = sorted(flat_record)
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_fields)
                writer.writeheader()

        row = {field: flat_record.get(field) for field in self._csv_fields}
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_fields)
            writer.writerow(row)


def _flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_record(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = ";".join(str(item) for item in value)
        else:
            flat[name] = value
    return flat

