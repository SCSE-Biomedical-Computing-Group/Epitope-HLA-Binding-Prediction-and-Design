#!/usr/bin/env python3
"""
Shared evaluation metrics for generation experiments.
"""

from itertools import combinations
from typing import Dict, Iterable, List
import numpy as np
from generator.selection.diversity import edit_distance, get_kmers


def scores_to_affinity(scores: Iterable[float]) -> np.ndarray:
    """Convert -log10(nM) scores to affinity values in nM."""
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)

    affinities = []
    for score in arr:
        try:
            affinity = 10 ** (-score)
        except (OverflowError, ValueError):
            affinity = 50000.0
        if not np.isfinite(affinity):
            affinity = 50000.0
        affinities.append(float(affinity))

    return np.asarray(affinities, dtype=float)


def binding_rates_from_scores(scores: Iterable[float]) -> Dict[str, float]:
    """Compute evaluation-facing binder rates from predictor scores."""
    scores_arr = np.asarray(list(scores), dtype=float)
    affinities_nM = scores_to_affinity(scores_arr)
    n = len(affinities_nM)
    if n == 0:
        return {
            "n": 0,
            "pct_SB": 0.0,
            "pct_WB": 0.0,
            "pct_IEDB_500nM": 0.0,
            "median_nM": 0.0,
            "mean_score": 0.0,
        }

    sb = int(np.sum(affinities_nM <= 50))
    wb = int(np.sum(affinities_nM <= 500))
    iedb = int(np.sum(affinities_nM <= 500))
    return {
        "n": n,
        "pct_SB": sb / n * 100,
        "pct_WB": wb / n * 100,
        "pct_IEDB_500nM": iedb / n * 100,
        "median_nM": float(np.median(affinities_nM)),
        "mean_score": float(np.mean(scores_arr)),
    }


def evaluate_binding_pass_rates(
    peptides: List[str],
    scores: List[float],
    allele: str,
) -> Dict[str, object]:
    """
    Evaluate generated peptides against standard binding thresholds.
    """
    results: Dict[str, object] = {"n": len(peptides)}
    if not scores:
        return results

    results.update(binding_rates_from_scores(scores))
    results["median_affinity_nM"] = float(results["median_nM"])

    n = int(results["n"])
    sb = int(round(float(results["pct_SB"]) * n / 100))
    wb = int(round(float(results["pct_WB"]) * n / 100))

    print(f"\n  Binding pass-rates for {allele} ({n} peptides):")
    print(f"     %SB  (IC50 <= 50 nM):   {results['pct_SB']:.1f}%  ({sb}/{n})")
    print(f"     %WB  (IC50 <= 500 nM):  {results['pct_WB']:.1f}%  ({wb}/{n})")
    print(
        "     Median affinity:        "
        f"{results['median_affinity_nM']:.1f} nM"
    )
    print(f"     Mean score (-log10 nM): {results['mean_score']:.3f}")

    return results


def diversity_stats(
    peptides: List[str],
    pair_sample_size: int = 200,
    kmer_size: int = 3,
) -> Dict[str, float]:
    """Compute diversity metrics with consistent settings."""
    if not peptides:
        return {
            "avg_edit_dist": 0.0,
            "min_edit_dist": 0,
            "unique_3mers": 0,
        }

    n = min(len(peptides), pair_sample_size)
    subset = peptides[:n]
    dists = [
        edit_distance(subset[i], subset[j])
        for i, j in combinations(range(n), 2)
    ]

    kmers = set()
    for pep in peptides:
        kmers.update(get_kmers(pep, k=kmer_size))

    return {
        "avg_edit_dist": float(np.mean(dists)) if dists else 0.0,
        "min_edit_dist": int(np.min(dists)) if dists else 0,
        "unique_3mers": int(len(kmers)),
    }
