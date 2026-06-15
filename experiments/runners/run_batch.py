"""Run a manifest of sketches through one experiment config."""
from __future__ import annotations

import argparse

from experiments.runners.common import load_config, load_manifest, make_logger, run_case


def run_batch(
    config_path: str,
    manifest_path: str,
    limit: int | None = None,
    prompt_only: bool = False,
):
    config = load_config(config_path)
    cases = load_manifest(manifest_path, limit=limit)
    logger = make_logger(config)

    records = []
    for case in cases:
        records.append(run_case(config, case, logger, prompt_only=prompt_only))

    print(f"Batch complete: {logger.run_id} ({len(records)} records)")
    print(f"CSV: {logger.csv_path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch experiment from a dataset manifest.")
    parser.add_argument("--config", default="experiments/configs/sd15.json")
    parser.add_argument("--manifest", default="datasets/manifests/pilot_41.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()
    run_batch(args.config, args.manifest, limit=args.limit, prompt_only=args.prompt_only)


if __name__ == "__main__":
    main()

