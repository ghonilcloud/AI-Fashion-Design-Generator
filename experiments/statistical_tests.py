"""Run paired statistical tests on experiment results.

Default input:
    experiments/results/raw_results.csv

Outputs:
    experiments/results/statistical_tests.csv
    experiments/results/statistical_tests.md
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


try:
    from scipy import stats
except Exception:
    stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "experiments" / "results" / "raw_results.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

PROMPT_A = "rule_based"
PROMPT_B = "llm"
ALPHA = 0.05
METRICS = ["clip_score", "lpips_score", "sketch_iou", "latency_seconds"]
PAIRING_KEYS = [
    "sketch_id",
    "model_key",
    "num_inference_steps",
    "guidance_scale",
    "controlnet_conditioning_scale",
    "seed",
]

CSV_FIELDS = [
    "metric",
    "comparison",
    "pairing_keys",
    "number_of_pairs",
    "rule_based_mean",
    "llm_mean",
    "mean_difference_llm_minus_rule_based",
    "normality_test",
    "normality_statistic",
    "normality_p_value",
    "normality_passed",
    "test_used",
    "test_statistic",
    "p_value",
    "significant_p_lt_0_05",
    "interpretation",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired statistical tests on raw experiment results.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_path)
    results = [analyze_metric(rows, metric) for metric in METRICS]
    write_csv(output_dir / "statistical_tests.csv", results)
    write_markdown(output_dir / "statistical_tests.md", results)
    print(f"Statistical test results written to {output_dir}")


def analyze_metric(rows: list[dict[str, str]], metric: str) -> dict[str, Any]:
    pairs = paired_values(rows, metric)
    rule_values = [pair[0] for pair in pairs]
    llm_values = [pair[1] for pair in pairs]
    differences = [llm - rule for rule, llm in pairs]
    notes: list[str] = []

    normality = normality_test(differences)
    notes.extend(normality.get("notes", []))

    if normality["normality_passed"]:
        test_result = paired_t_test(rule_values, llm_values)
    else:
        test_result = wilcoxon_signed_rank_test(rule_values, llm_values)
    notes.extend(test_result.get("notes", []))

    p_value = test_result.get("p_value")
    significant = p_value is not None and p_value < ALPHA
    interpretation = interpret_result(
        metric=metric,
        rule_values=rule_values,
        llm_values=llm_values,
        p_value=p_value,
        significant=significant,
    )

    return {
        "metric": metric,
        "comparison": f"{PROMPT_A} vs {PROMPT_B}",
        "pairing_keys": ";".join(PAIRING_KEYS),
        "number_of_pairs": len(pairs),
        "rule_based_mean": safe_mean(rule_values),
        "llm_mean": safe_mean(llm_values),
        "mean_difference_llm_minus_rule_based": safe_mean(differences),
        "normality_test": normality.get("test_used"),
        "normality_statistic": normality.get("statistic"),
        "normality_p_value": normality.get("p_value"),
        "normality_passed": normality.get("normality_passed"),
        "test_used": test_result.get("test_used"),
        "test_statistic": test_result.get("statistic"),
        "p_value": p_value,
        "significant_p_lt_0_05": significant,
        "interpretation": interpretation,
        "notes": " ".join(notes),
    }


def paired_values(rows: list[dict[str, str]], metric: str) -> list[tuple[float, float]]:
    grouped: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        prompt_strategy = row.get("prompt_strategy")
        if prompt_strategy not in {PROMPT_A, PROMPT_B}:
            continue

        value = parse_float(row.get(metric))
        if value is None:
            continue

        key = tuple(row.get(pairing_key, "") for pairing_key in PAIRING_KEYS)
        grouped[key][prompt_strategy] = value

    pairs = []
    for values in grouped.values():
        if PROMPT_A in values and PROMPT_B in values:
            pairs.append((values[PROMPT_A], values[PROMPT_B]))
    return pairs


def normality_test(differences: list[float]) -> dict[str, Any]:
    if differences and max(differences) - min(differences) < 1e-12:
        return {
            "test_used": "constant_differences",
            "statistic": None,
            "p_value": None,
            "normality_passed": False,
            "notes": ["Normality was not tested because all paired differences were constant."],
        }

    if len(differences) < 3:
        return {
            "test_used": "not_enough_pairs",
            "statistic": None,
            "p_value": None,
            "normality_passed": False,
            "notes": ["Normality was not tested because fewer than three paired differences were available."],
        }

    if stats is not None:
        if len(differences) <= 5000:
            result = stats.shapiro(differences)
            return {
                "test_used": "Shapiro-Wilk",
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "normality_passed": bool(result.pvalue >= ALPHA),
                "notes": [],
            }

        result = stats.normaltest(differences)
        return {
            "test_used": "D'Agostino K^2",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "normality_passed": bool(result.pvalue >= ALPHA),
            "notes": [],
        }

    statistic, p_value = jarque_bera_fallback(differences)
    return {
        "test_used": "Jarque-Bera fallback",
        "statistic": statistic,
        "p_value": p_value,
        "normality_passed": bool(p_value >= ALPHA),
        "notes": ["SciPy was unavailable, so a Jarque-Bera fallback normality test was used."],
    }


def paired_t_test(rule_values: list[float], llm_values: list[float]) -> dict[str, Any]:
    if len(rule_values) < 2:
        return {
            "test_used": "paired t-test",
            "statistic": None,
            "p_value": None,
            "notes": ["Paired t-test requires at least two pairs."],
        }

    if stats is not None:
        result = stats.ttest_rel(llm_values, rule_values, nan_policy="omit")
        return {
            "test_used": "paired t-test",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "notes": [],
        }

    differences = [llm - rule for rule, llm in zip(rule_values, llm_values)]
    sd = stdev(differences)
    if sd == 0:
        return {
            "test_used": "paired t-test fallback",
            "statistic": 0.0,
            "p_value": 1.0,
            "notes": ["SciPy was unavailable; all paired differences were identical."],
        }
    statistic = mean(differences) / (sd / math.sqrt(len(differences)))
    p_value = two_sided_normal_p_value(statistic)
    return {
        "test_used": "paired t-test fallback",
        "statistic": statistic,
        "p_value": p_value,
        "notes": ["SciPy was unavailable, so the paired t-test used a normal-approximation fallback."],
    }


def wilcoxon_signed_rank_test(rule_values: list[float], llm_values: list[float]) -> dict[str, Any]:
    if not rule_values:
        return {
            "test_used": "Wilcoxon signed-rank test",
            "statistic": None,
            "p_value": None,
            "notes": ["Wilcoxon test requires at least one pair."],
        }

    differences = [llm - rule for rule, llm in zip(rule_values, llm_values)]
    if all(abs(diff) < 1e-12 for diff in differences):
        return {
            "test_used": "Wilcoxon signed-rank test",
            "statistic": 0.0,
            "p_value": 1.0,
            "notes": ["All paired differences were zero."],
        }

    if stats is not None:
        result = stats.wilcoxon(llm_values, rule_values, zero_method="wilcox", alternative="two-sided")
        return {
            "test_used": "Wilcoxon signed-rank test",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "notes": [],
        }

    statistic, p_value = wilcoxon_fallback(differences)
    return {
        "test_used": "Wilcoxon signed-rank fallback",
        "statistic": statistic,
        "p_value": p_value,
        "notes": ["SciPy was unavailable, so the Wilcoxon test used a normal-approximation fallback."],
    }


def interpret_result(
    metric: str,
    rule_values: list[float],
    llm_values: list[float],
    p_value: float | None,
    significant: bool,
) -> str:
    if p_value is None:
        return f"There were not enough matched pairs to test {metric}."

    rule_mean = safe_mean(rule_values)
    llm_mean = safe_mean(llm_values)
    if rule_mean is None or llm_mean is None:
        return f"There were not enough numeric values to interpret {metric}."

    if metric in {"clip_score", "sketch_iou"}:
        better_group = PROMPT_B if llm_mean > rule_mean else PROMPT_A if rule_mean > llm_mean else "neither strategy"
        direction_note = "higher is better"
    else:
        better_group = PROMPT_B if llm_mean < rule_mean else PROMPT_A if rule_mean < llm_mean else "neither strategy"
        direction_note = "lower is better"

    if significant:
        return (
            f"For {metric}, the difference between {PROMPT_A} and {PROMPT_B} is statistically "
            f"significant at p < 0.05. Based on the group means, {better_group} performs better "
            f"for this metric ({direction_note})."
        )

    return (
        f"For {metric}, the observed difference between {PROMPT_A} and {PROMPT_B} is not "
        f"statistically significant at p < 0.05, so the experiment does not provide strong "
        f"evidence that one prompt strategy outperforms the other for this metric."
    )


def write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field)) for field in CSV_FIELDS})


def write_markdown(output_path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Statistical Tests",
        "",
        f"Comparison: `{PROMPT_A}` vs `{PROMPT_B}` prompt strategies.",
        "",
        "Pairs were matched using:",
        "",
        ", ".join(f"`{key}`" for key in PAIRING_KEYS),
        "",
        "Normality was tested on paired differences (`llm - rule_based`). If paired differences "
        "were normally distributed, a paired t-test was used. Otherwise, a Wilcoxon signed-rank "
        "test was used.",
        "",
        "| Metric | Groups Compared | Test Used | Normality p-value | p-value | Significant at p < 0.05 | Interpretation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["metric"]),
                    str(row["comparison"]),
                    str(row["test_used"]),
                    format_cell(row["normality_p_value"]),
                    format_cell(row["p_value"]),
                    "Yes" if row["significant_p_lt_0_05"] else "No",
                    str(row["interpretation"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Metric Interpretation Notes",
            "",
            "- CLIP score: higher is better.",
            "- LPIPS score: lower is better.",
            "- Sketch IoU: higher is better.",
            "- Latency: lower is better for deployment.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def read_rows(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Raw results CSV not found: {input_path}")
    with input_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def jarque_bera_fallback(values: list[float]) -> tuple[float, float]:
    n = len(values)
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / n
    if variance == 0:
        return 0.0, 1.0
    std = math.sqrt(variance)
    skewness = sum(((value - avg) / std) ** 3 for value in values) / n
    kurtosis = sum(((value - avg) / std) ** 4 for value in values) / n
    statistic = n / 6 * (skewness**2 + ((kurtosis - 3) ** 2) / 4)
    p_value = math.exp(-statistic / 2)
    return statistic, p_value


def wilcoxon_fallback(differences: list[float]) -> tuple[float, float]:
    non_zero = [diff for diff in differences if abs(diff) >= 1e-12]
    ranked = sorted((abs(diff), 1 if diff > 0 else -1) for diff in non_zero)
    positive_rank_sum = 0.0
    negative_rank_sum = 0.0
    for rank, (_, sign) in enumerate(ranked, start=1):
        if sign > 0:
            positive_rank_sum += rank
        else:
            negative_rank_sum += rank
    statistic = min(positive_rank_sum, negative_rank_sum)
    n = len(ranked)
    expected = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    if variance == 0:
        return statistic, 1.0
    z_value = (statistic - expected) / math.sqrt(variance)
    return statistic, two_sided_normal_p_value(z_value)


def two_sided_normal_p_value(statistic: float) -> float:
    return math.erfc(abs(statistic) / math.sqrt(2))


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


if __name__ == "__main__":
    main()
