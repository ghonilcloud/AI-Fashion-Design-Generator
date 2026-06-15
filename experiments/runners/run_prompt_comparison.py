"""Compare rule-based and Gemini prompt strategies."""
from __future__ import annotations

import argparse

from experiments.runners.common import (
    clone_config_with_overrides,
    load_config,
    load_manifest,
    make_logger,
    run_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt baseline versus Gemini comparison.")
    parser.add_argument(
        "--comparison-config",
        default="experiments/configs/prompt_baseline_vs_gemini.json",
    )
    parser.add_argument("--manifest", default="datasets/manifests/pilot_41.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()

    comparison = load_config(args.comparison_config)
    base_config = load_config(comparison["base_config"])
    cases = load_manifest(args.manifest, limit=args.limit)

    for prompt_strategy in comparison.get("prompt_strategies", []):
        config = clone_config_with_overrides(
            base_config,
            experiment_name=f"{comparison['experiment_name']}_{prompt_strategy}",
            prompt__prompt_strategy=prompt_strategy,
        )
        logger = make_logger(config)
        for case in cases:
            run_case(config, case, logger, prompt_only=args.prompt_only)
        print(f"Prompt strategy {prompt_strategy} complete: {logger.run_id}")


if __name__ == "__main__":
    main()
