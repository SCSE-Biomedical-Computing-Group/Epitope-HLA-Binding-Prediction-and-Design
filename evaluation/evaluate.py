#!/usr/bin/env python3
"""
Unified evaluation script for generation experiments.
"""

import csv
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
_metrics = importlib.import_module("evaluation.metrics")
binding_rates_from_scores = _metrics.binding_rates_from_scores
diversity_stats = _metrics.diversity_stats


METRIC_KEYS = [
    "pct_SB",
    "pct_WB",
    "median_nM",
    "avg_edit_dist",
    "unique_3mers",
]


def allele_to_dir(allele: str) -> str:
    """Convert an allele string into the output directory format."""
    return allele.replace("*", "_").replace(":", "_")


def resolve_alleles(alleles_arg: str, reference_run: Path) -> List[str]:
    """Resolve the allele list from CLI input or a reference run summary."""
    if alleles_arg.strip().lower() == "all":
        summary_path = reference_run / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"run_summary.json not found in {reference_run}"
            )

        with open(summary_path) as file_handle:
            payload = json.load(file_handle)

        allele_rows = payload.get("allele_summaries", [])
        alleles = [
            row.get("allele")
            for row in allele_rows
            if row.get("allele")
        ]
        if not alleles:
            raise ValueError(f"No alleles found in {summary_path}")
        return alleles

    return [a.strip() for a in alleles_arg.split(",") if a.strip()]


def load_condition_results(
    run_dir: Path,
    alleles: List[str],
    csv_name: str = "final_peptides_div.csv",
) -> tuple[Dict[str, pd.DataFrame], Dict]:
    """Load per-allele result tables for a condition run directory."""
    results = {}
    loaded_files = {}
    missing_alleles = []
    for allele in alleles:
        adir = run_dir / allele_to_dir(allele)
        csv_path = adir / csv_name
        if csv_path.exists():
            results[allele] = pd.read_csv(csv_path)
            loaded_files[allele] = str(csv_path)
            continue

        # Baseline runs use a different csv filename.
        alt = adir / "baseline_peptides.csv"
        if alt.exists():
            results[allele] = pd.read_csv(alt)
            loaded_files[allele] = str(alt)
            continue

        legacy_alt = adir / "random_peptides.csv"
        if legacy_alt.exists():
            results[allele] = pd.read_csv(legacy_alt)
            loaded_files[allele] = str(legacy_alt)
            continue

        missing_alleles.append(allele)

    return results, {
        "requested_alleles": alleles,
        "loaded_alleles": sorted(results),
        "missing_alleles": missing_alleles,
        "loaded_files": loaded_files,
        "num_requested": len(alleles),
        "num_loaded": len(results),
        "num_missing": len(missing_alleles),
    }


def evaluate_condition(
    label: str,
    allele_dfs: Dict[str, pd.DataFrame],
) -> Dict:
    """Compute per-allele and macro-average metrics for one condition."""
    per_allele = {}
    all_sb, all_wb, all_med, all_div, all_3mer = [], [], [], [], []

    for allele, df in allele_dfs.items():
        scores = df["score"].values
        rates = binding_rates_from_scores(scores)
        peptides = df["peptide"].tolist()
        dstats = diversity_stats(peptides)

        extra = {}
        if "score_improvement" in df.columns:
            values = [
                float(v)
                for v in df["score_improvement"].dropna().tolist()
            ]
            if values:
                extra["num_improved"] = int(sum(1 for v in values if v > 0))
                extra["avg_score_improvement"] = float(np.mean(values))

        per_allele[allele] = {**rates, **dstats, **extra}
        all_sb.append(rates["pct_SB"])
        all_wb.append(rates["pct_WB"])
        all_med.append(rates["median_nM"])
        all_div.append(dstats["avg_edit_dist"])
        all_3mer.append(dstats["unique_3mers"])

    return {
        "label": label,
        "per_allele": per_allele,
        "macro_avg": {
            "pct_SB": float(np.mean(all_sb)) if all_sb else 0.0,
            "pct_WB": float(np.mean(all_wb)) if all_wb else 0.0,
            "median_nM": float(np.mean(all_med)) if all_med else 0.0,
            "avg_edit_dist": float(np.mean(all_div)) if all_div else 0.0,
            "unique_3mers": float(np.mean(all_3mer)) if all_3mer else 0.0,
        },
    }


def label_to_key(label: str) -> str:
    """Normalize a condition label into a filesystem-safe key."""
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in label
    ).strip("_")


def summarize_novelty(novelty_csv: Path | None) -> Dict:
    """Summarize novelty rates from an optional novelty CSV file."""
    if novelty_csv is None:
        return {}
    if not novelty_csv.exists():
        return {"path": str(novelty_csv), "exists": False}

    with open(novelty_csv) as handle:
        rows = list(csv.DictReader(handle))

    total = len(rows)
    exact = sum(
        1 for r in rows if str(r.get("is_novel", "")).lower() == "true"
    )
    summary = {
        "path": str(novelty_csv),
        "exists": True,
        "total": total,
        "exact_novel": exact,
        "exact_novel_rate": (exact / total * 100) if total else 0.0,
    }

    if rows and "is_strict_novel" in rows[0]:
        strict = sum(
            1
            for r in rows
            if str(r.get("is_strict_novel", "")).lower() == "true"
        )
        summary["strict_novel"] = strict
        summary["strict_novel_rate"] = (strict / total * 100) if total else 0.0

    return summary


def normalize_condition_specs(
    condition_specs: List[tuple[str, Path]],
) -> List[tuple[str, Path]]:
    """Validate and normalize repeated ``--condition`` CLI inputs."""
    if not condition_specs:
        raise click.UsageError(
            "Provide at least one --condition LABEL PATH pair."
        )

    normalized = []
    seen_labels = set()
    for label, run_dir in condition_specs:
        cleaned_label = label.strip()
        if not cleaned_label:
            raise click.UsageError("Condition labels must be non-empty.")
        if cleaned_label in seen_labels:
            raise click.UsageError(
                f"Duplicate condition label: {cleaned_label}"
            )
        seen_labels.add(cleaned_label)
        normalized.append((cleaned_label, run_dir))
    return normalized


def run_single(args):
    """Evaluate one set of named condition runs and write a JSON summary."""
    condition_specs = normalize_condition_specs(args.conditions)
    reference_label, reference_run = condition_specs[0]
    allele_source = (
        "reference_run_summary"
        if args.alleles.strip().lower() == "all"
        else "cli"
    )
    alleles = resolve_alleles(args.alleles, reference_run)

    conditions = []
    condition_runs = []
    for label, run_dir in condition_specs:
        allele_dfs, load_report = load_condition_results(run_dir, alleles)
        conditions.append(
            evaluate_condition(
                label,
                allele_dfs,
            )
        )
        condition_runs.append(
            {
                "label": label,
                "path": str(run_dir),
                "load_report": load_report,
            }
        )

    novelty_summary = summarize_novelty(args.novelty_csv)
    deltas = {}
    if conditions:
        reference_eval = conditions[0]
        for comparison_eval in conditions[1:]:
            shared_alleles = sorted(
                set(reference_eval["per_allele"])
                & set(comparison_eval["per_allele"])
            )
            per_allele_delta = {}
            for allele in shared_alleles:
                reference_row = reference_eval["per_allele"][allele]
                comparison_row = comparison_eval["per_allele"][allele]
                per_allele_delta[allele] = {
                    metric: reference_row[metric] - comparison_row[metric]
                    for metric in METRIC_KEYS
                }

            delta = {
                "reference": reference_eval["label"],
                "comparison": comparison_eval["label"],
                "shared_alleles": shared_alleles,
                "per_allele": per_allele_delta,
                "macro_avg": {
                    metric: (
                        float(
                            np.mean(
                                [
                                    row[metric]
                                    for row in per_allele_delta.values()
                                ]
                            )
                        )
                        if per_allele_delta else 0.0
                    )
                    for metric in METRIC_KEYS
                },
            }
            delta_key = (
                f"{label_to_key(delta['reference'])}"
                f"_minus_{label_to_key(delta['comparison'])}"
            )
            deltas[delta_key] = delta

    payload = {
        "mode": "run",
        "reference_condition": reference_label,
        "allele_source": allele_source,
        "condition_runs": condition_runs,
        "alleles": alleles,
        "conditions": {c["label"]: c for c in conditions},
        "deltas": deltas,
        "novelty": novelty_summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, default=str)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Evaluate generation runs",
)
@click.option(
    "--condition",
    "conditions",
    type=(str, click.Path(path_type=Path)),
    multiple=True,
    help=(
        "Condition label and run directory. Repeat this option for each run; "
        "the first condition is treated as the reference."
    ),
)
@click.option("--alleles", default="all", show_default=True)
@click.option(
    "--novelty_csv",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("evaluation/comparison_results.json"),
    show_default=True,
)
def main(
    conditions: tuple[tuple[str, Path], ...],
    alleles: str,
    novelty_csv: Path | None,
    out: Path,
) -> None:
    """Evaluate one or more named condition runs."""
    run_single(
        SimpleNamespace(
            conditions=list(conditions),
            alleles=alleles,
            novelty_csv=novelty_csv,
            out=out,
        )
    )


if __name__ == "__main__":
    main()
