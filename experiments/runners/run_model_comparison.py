"""Run multiple model configs on the same manifest."""
from __future__ import annotations

import argparse

from experiments.runners.common import load_config
from experiments.runners.run_batch import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model comparison configs.")
    parser.add_argument("--comparison-config", default="experiments/configs/model_comparison.json")
    parser.add_argument("--manifest", default="datasets/manifests/pilot_41.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()

    comparison = load_config(args.comparison_config)
    for config_path in comparison.get("configs", []):
        run_batch(config_path, args.manifest, limit=args.limit, prompt_only=args.prompt_only)


if __name__ == "__main__":
    main()

