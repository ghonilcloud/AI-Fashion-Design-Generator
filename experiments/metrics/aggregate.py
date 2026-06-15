"""Aggregate CSV experiment records into mean/std summaries."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


def aggregate_csv(csv_path: str | Path) -> dict[str, dict[str, float]]:
    numeric_values: dict[str, list[float]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    numeric_values.setdefault(key, []).append(float(value))
                except ValueError:
                    continue

    summary: dict[str, dict[str, float]] = {}
    for key, values in numeric_values.items():
        summary[key] = {
            "count": float(len(values)),
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate numeric experiment CSV columns.")
    parser.add_argument("csv_path")
    args = parser.parse_args()
    for key, stats in aggregate_csv(args.csv_path).items():
        print(f"{key}: n={int(stats['count'])}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")


if __name__ == "__main__":
    main()

