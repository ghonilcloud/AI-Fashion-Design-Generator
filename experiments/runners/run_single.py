"""Run one sketch through one experiment config."""
from __future__ import annotations

import argparse

from experiments.runners.common import clone_config_with_overrides, load_config, make_logger, run_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fashion generation experiment case.")
    parser.add_argument("--config", default="experiments/configs/sd15.json")
    parser.add_argument("--sketch", required=True)
    parser.add_argument("--sketch-id", default="single")
    parser.add_argument("--garment-type", default="fashion garment")
    parser.add_argument("--tones", default="")
    parser.add_argument("--kansei-words", default="")
    parser.add_argument("--manual-prompt", default="")
    parser.add_argument("--prompt-strategy", choices=["llm", "rule_based"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prompt_strategy:
        config = clone_config_with_overrides(config, prompt__prompt_strategy=args.prompt_strategy)
    if args.seed is not None:
        config = clone_config_with_overrides(config, generation__seed=args.seed)

    case = {
        "sketch_id": args.sketch_id,
        "sketch_path": args.sketch,
        "garment_type": args.garment_type,
        "tones": _split_cli_list(args.tones),
        "kansei_words": _split_cli_list(args.kansei_words),
        "manual_prompt": args.manual_prompt,
    }

    logger = make_logger(config, suffix=args.sketch_id)
    record = run_case(config, case, logger, prompt_only=args.prompt_only)
    print(f"Run {record['status']}: {logger.run_id}")


def _split_cli_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


if __name__ == "__main__":
    main()
