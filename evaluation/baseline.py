#!/usr/bin/env python3
"""
Random peptide baseline generator.

Generates peptides by uniform sampling from the standard amino acid alphabet.
Scores them with NetMHCpan to provide a lower-bound baseline for comparison
with the pipeline.
"""

from contextlib import redirect_stderr, redirect_stdout
import traceback
import random
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from explainer.shap import netmhcpan_score_batch
from evaluation.metrics import (
    binding_rates_from_scores,
    diversity_stats,
    scores_to_affinity,
)

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
LENGTH_QUOTAS = {8: 0.10, 9: 0.40, 10: 0.30, 11: 0.20}


def generate_random_peptides(
    n: int,
    length_dist: Dict[int, float] | None = None,
    seed: int = 42,
) -> List[str]:
    """Generate n random peptides according to length distribution."""
    rng = random.Random(seed)
    if length_dist is None:
        length_dist = LENGTH_QUOTAS

    peptides = set()
    lengths = list(length_dist.keys())
    weights = list(length_dist.values())

    while len(peptides) < n:
        L = rng.choices(lengths, weights=weights, k=1)[0]
        pep = "".join(rng.choices(STANDARD_AAS, k=L))
        peptides.add(pep)

    return sorted(peptides)


def score_peptides(
    peptides: List[str],
    allele: str,
    netmhcpan_path: Path,
) -> np.ndarray:
    """Score peptides via NetMHCpan, returns -log10(nM) scores."""
    return netmhcpan_score_batch(peptides, allele, netmhcpan_path)


def compute_stats(peptides: List[str], scores: np.ndarray) -> Dict:
    """Compute binding and diversity stats for a set of scored peptides."""
    rates = binding_rates_from_scores(scores)
    dstats = diversity_stats(peptides)

    length_dist = {}
    for p in peptides:
        L = len(p)
        length_dist[L] = length_dist.get(L, 0) + 1

    return {
        "n": rates["n"],
        "pct_SB": rates["pct_SB"],
        "pct_WB": rates["pct_WB"],
        "median_affinity_nM": rates["median_nM"],
        "mean_score": rates["mean_score"],
        "std_score": float(np.std(scores)),
        "avg_edit_distance": dstats["avg_edit_dist"],
        "min_edit_distance": dstats["min_edit_dist"],
        "unique_3mers": dstats["unique_3mers"],
        "length_distribution": length_dist,
    }


def load_alleles_from_run_summary(run_dir: Path) -> List[str]:
    """Load allele list from generation run summary."""
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"run_summary.json not found in {run_dir}")
    with open(summary_path) as f:
        payload = json.load(f)
    allele_rows = payload.get("allele_summaries", [])
    alleles = [row.get("allele") for row in allele_rows if row.get("allele")]
    if not alleles:
        raise ValueError(f"No alleles found in {summary_path}")
    return alleles


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Peptide baseline",
)
@click.option(
    "--alleles",
    default="all",
    show_default=True,
    help='Comma-separated alleles, or "all"',
)
@click.option(
    "--reference_run",
    type=click.Path(path_type=Path),
    default=None,
    help='Run directory used to resolve alleles when --alleles is "all"',
)
@click.option(
    "--num",
    type=int,
    default=100,
    show_default=True,
    help="Number of peptides per allele",
)
@click.option(
    "--netmhcpan",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to NetMHCpan 4.2 installation",
)
@click.option(
    "--out_dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/baseline"),
    show_default=True,
    help="Output directory",
)
@click.option("--seed", type=int, default=42, show_default=True)
def main(
    alleles: str,
    reference_run: Path | None,
    num: int,
    netmhcpan: Path,
    out_dir: Path,
    seed: int,
) -> None:
    """Generate and score random peptides as a baseline."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = out_dir / run_id
    run_out_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_out_dir / "run.log"

    with open(log_path, "a", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            try:
                _run_baseline(
                    alleles=alleles,
                    reference_run=reference_run,
                    num=num,
                    netmhcpan=netmhcpan,
                    run_out_dir=run_out_dir,
                    seed=seed,
                )
            except Exception:
                traceback.print_exc()
                raise SystemExit(1) from None


def _run_baseline(
    *,
    alleles: str,
    reference_run: Path | None,
    num: int,
    netmhcpan: Path,
    run_out_dir: Path,
    seed: int,
) -> None:
    """Execute the random baseline generation workflow."""
    if alleles.strip().lower() == "all":
        if reference_run is None:
            raise click.UsageError(
                '--reference_run is required when --alleles is "all"'
            )
        allele_list = load_alleles_from_run_summary(reference_run)
        print(f"Resolved {len(allele_list)} alleles from {reference_run}")
    else:
        allele_list = [a.strip() for a in alleles.split(",") if a.strip()]

    all_stats = []

    for allele in allele_list:
        print(f"\n{'='*50}")
        print(f"Random baseline for {allele}")
        print("=" * 50)

        peptides = generate_random_peptides(num, seed=seed)
        print(f"  Generated {len(peptides)} random peptides")

        print("  Scoring with NetMHCpan...")
        scores = score_peptides(peptides, allele, netmhcpan)
        print(f"  Mean score: {np.mean(scores):.4f}")

        stats = compute_stats(peptides, scores)
        stats["allele"] = allele
        all_stats.append(stats)

        print(f"  %SB:  {stats['pct_SB']:.1f}%")
        print(f"  %WB:  {stats['pct_WB']:.1f}%")
        print(f"  Median affinity: {stats['median_affinity_nM']:.1f} nM")

        # Save CSV
        allele_dir = run_out_dir / allele.replace("*", "_").replace(":", "_")
        allele_dir.mkdir(parents=True, exist_ok=True)
        affinities = scores_to_affinity(scores).tolist()
        df = pd.DataFrame({
            "allele": allele,
            "peptide": peptides,
            "length": [len(p) for p in peptides],
            "score": scores,
            "affinity_nM": affinities,
            "source": "random",
        })
        df.to_csv(allele_dir / "baseline_peptides.csv", index=False)

    # Save summary
    with open(run_out_dir / "baseline_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*50}")
    print("Random baseline summary")
    print("=" * 50)
    for s in all_stats:
        print(f"  {s['allele']}: %SB={s['pct_SB']:.1f}%  %WB={s['pct_WB']:.1f}%  "
              f"median={s['median_affinity_nM']:.0f} nM")
    print(f"\nSaved to {run_out_dir}")


if __name__ == "__main__":
    main()
