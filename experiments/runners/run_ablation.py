"""Run ControlNet conditioning-scale ablations."""
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
    parser = argparse.ArgumentParser(description="Run ControlNet scale ablation.")
    parser.add_argument("--ablation-config", default="experiments/configs/ablation_controlnet_scale.json")
    parser.add_argument("--manifest", default="datasets/manifests/pilot_41.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()

    ablation = load_config(args.ablation_config)
    base_config = load_config(ablation["base_config"])
    cases = load_manifest(args.manifest, limit=args.limit)

    for scale in ablation.get("controlnet_conditioning_scales", []):
        config = clone_config_with_overrides(
            base_config,
            experiment_name=f"{ablation['experiment_name']}_scale_{scale}",
            generation__controlnet_conditioning_scale=scale,
        )
        logger = make_logger(config)
        for case in cases:
            run_case(config, case, logger, prompt_only=args.prompt_only)
        print(f"Ablation scale {scale} complete: {logger.run_id}")


if __name__ == "__main__":
    main()

