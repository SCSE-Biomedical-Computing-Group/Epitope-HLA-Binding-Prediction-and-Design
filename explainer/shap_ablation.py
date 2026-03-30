#!/usr/bin/env python3
"""
SHAP ablation analysis utilities for the peptide-generation pipeline.

Primary mode:
    pipeline — compares:
        Full:     generator -> predictor scoring -> SHAP-guided refinement
                  -> predictor rescoring -> selection
        Ablation: generator -> predictor scoring -> selection (no SHAP step)

Optional SHAP ablation analysis modes (legacy):
    v1       — basic SHAP top-k / bottom-k / random-k mutation analysis
    enhanced — stratified SHAP ablation analysis + correlation analysis
"""

from contextlib import redirect_stderr, redirect_stdout
import json
import random
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
from scipy import stats

# Allow running this file directly (python explainer/shap_ablation.py).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainer.shap import (
    AA_TO_IDX,
    PEP_LENGTHS,
    STANDARD_AAS,
    netmhcpan_score_batch,
)

# ============================================================================
# CONFIG
# ============================================================================
N_MUTATIONS = 20       # random mutations to average per position
K_VALUES = [1, 2, 3]   # ablation depths
N_SAMPLES = 2000       # samples to validate per allele

# AA property groups for controlled mutations
HYDROPHOBIC = set("FILMVWY")
CHARGED = set("DEKR")
POLAR = set("CHNQST")
SMALL = set("AG")


def _non_preferred_aa(aa: str) -> str:
    """Return a disruptive substitution for the given AA."""
    if aa in HYDROPHOBIC:
        return "D"
    elif aa in CHARGED:
        return "L"
    return "D"


# ============================================================================
# MUTATION PRIMITIVES
# ============================================================================

def mutate_random(peptide: str, position: int, n_mutations: int = N_MUTATIONS) -> List[str]:
    """Replace position with random AAs; return list of mutated peptides."""
    results = []
    for _ in range(n_mutations):
        candidates = [aa for aa in STANDARD_AAS if aa != peptide[position]]
        new_aa = random.choice(candidates) if candidates else "A"
        results.append(peptide[:position] + new_aa + peptide[position + 1:])
    return results


def mutate_alanine(peptide: str, position: int) -> str:
    aa = "G" if peptide[position] == "A" else "A"
    return peptide[:position] + aa + peptide[position + 1:]


def mutate_controlled(peptide: str, position: int) -> str:
    return peptide[:position] + _non_preferred_aa(peptide[position]) + peptide[position + 1:]


def mutate_background(peptide: str, position: int) -> str:
    """Replace position with a uniformly random AA (SHAP-style perturbation).

    This most closely mirrors KernelSHAP's masking operator, which replaces
    features with values drawn from the background distribution.  When the
    background set is diverse, each AA is roughly equiprobable at each position.
    """
    new_aa = random.choice(STANDARD_AAS)
    return peptide[:position] + new_aa + peptide[position + 1:]


# ============================================================================
# ABLATION IMPACT (NETMHCPAN-BASED)
# ============================================================================

def compute_ablation_impact(
    peptides: List[str],
    allele: str,
    position: int,
    method: str = "random",
    netmhcpan_path: Optional[Path] = None,
    original_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute NetMHCpan score drop when *position* is mutated.

    Score convention: ``netmhcpan_score_batch`` returns −log10(nM), so
    higher = stronger binder.  A **positive** delta means the mutation
    *weakened* binding (expected when ablating an important position).

    Args:
        original_scores: Pre-computed −log10(nM) scores for *peptides*.
            When the caller already scored the full peptide set (e.g. once
            per ablation run), pass them here to avoid redundant NetMHCpan
            invocations.

    Returns:
        (N,) array of (original_score − ablated_score) per peptide.
    """
    if original_scores is None:
        original_scores = netmhcpan_score_batch(peptides, allele, netmhcpan_path)

    if method == "random":
        all_ablated = np.zeros(len(peptides))
        for _ in range(N_MUTATIONS):
            mutated = []
            for pep in peptides:
                candidates = [aa for aa in STANDARD_AAS if aa != pep[position]]
                new_aa = random.choice(candidates) if candidates else "A"
                mutated.append(pep[:position] + new_aa + pep[position + 1:])
            ablated_scores = netmhcpan_score_batch(mutated, allele, netmhcpan_path)
            all_ablated += ablated_scores
        ablated_mean = all_ablated / N_MUTATIONS
    elif method == "alanine":
        mutated = [mutate_alanine(p, position) for p in peptides]
        ablated_mean = netmhcpan_score_batch(mutated, allele, netmhcpan_path)
    elif method == "controlled":
        mutated = [mutate_controlled(p, position) for p in peptides]
        ablated_mean = netmhcpan_score_batch(mutated, allele, netmhcpan_path)
    elif method == "background":
        # SHAP-style: replace with uniformly random AA (mirrors KernelSHAP masking)
        all_ablated = np.zeros(len(peptides))
        for _ in range(N_MUTATIONS):
            mutated = [mutate_background(p, position) for p in peptides]
            ablated_scores = netmhcpan_score_batch(mutated, allele, netmhcpan_path)
            all_ablated += ablated_scores
        ablated_mean = all_ablated / N_MUTATIONS
    else:
        raise ValueError(f"Unknown ablation method: {method}")

    return original_scores - ablated_mean


# ============================================================================
# POSITION RANKING BY SHAP
# ============================================================================

def rank_positions_by_shap(
    shap_values: np.ndarray,
    peptides: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank peptide positions by |SHAP|.

    Args:
        shap_values: (N, L, 20) SHAP values
        peptides:    list of N peptides (same length L)

    Returns:
        ranked: (N, L) position indices sorted by descending |SHAP|
        valid_lengths: (N,) actual peptide lengths
    """
    N, L, _ = shap_values.shape

    pos_importance = np.zeros((N, L))
    for i, pep in enumerate(peptides):
        for j, aa_char in enumerate(pep):
            aa_idx = AA_TO_IDX.get(aa_char, 0)
            pos_importance[i, j] = abs(shap_values[i, j, aa_idx])

    ranked = np.argsort(-pos_importance, axis=1)
    valid_lengths = np.array([len(p) for p in peptides])
    return ranked, valid_lengths


# ============================================================================
# TOP-K / BOTTOM-K / RANDOM-K ABLATION
# ============================================================================

def ablate_top_k(
    peptides: List[str],
    allele: str,
    shap_values: np.ndarray,
    k: int,
    method: str = "random",
    netmhcpan_path: Optional[Path] = None,
) -> np.ndarray:
    """Ablate the top-k most important positions and return total score drop."""
    ranked, valid_lengths = rank_positions_by_shap(shap_values, peptides)
    # Cache original scores once for the whole peptide set
    all_original = netmhcpan_score_batch(peptides, allele, netmhcpan_path)
    total_delta = np.zeros(len(peptides))

    for ki in range(k):
        positions = ranked[:, ki]
        for pos in np.unique(positions):
            mask = positions == pos
            idx = np.where(mask)[0]
            valid_idx = [i for i in idx if pos < valid_lengths[i]]
            if not valid_idx:
                continue
            sub_peptides = [peptides[i] for i in valid_idx]
            sub_original = all_original[np.array(valid_idx)]
            deltas = compute_ablation_impact(
                sub_peptides, allele, int(pos), method, netmhcpan_path,
                original_scores=sub_original,
            )
            for j, vi in enumerate(valid_idx):
                total_delta[vi] += deltas[j]

    return total_delta


def ablate_bottom_k(
    peptides: List[str],
    allele: str,
    shap_values: np.ndarray,
    k: int,
    method: str = "random",
    netmhcpan_path: Optional[Path] = None,
) -> np.ndarray:
    """Ablate the bottom-k least important positions."""
    ranked, valid_lengths = rank_positions_by_shap(shap_values, peptides)
    # Cache original scores once for the whole peptide set
    all_original = netmhcpan_score_batch(peptides, allele, netmhcpan_path)
    total_delta = np.zeros(len(peptides))

    for ki in range(k):
        positions = np.array([
            ranked[i, valid_lengths[i] - 1 - ki] if ki < valid_lengths[i] else 0
            for i in range(len(peptides))
        ])
        for pos in np.unique(positions):
            mask = positions == pos
            idx = np.where(mask)[0]
            valid_idx = [i for i in idx if pos < valid_lengths[i]]
            if not valid_idx:
                continue
            sub_peptides = [peptides[i] for i in valid_idx]
            sub_original = all_original[np.array(valid_idx)]
            deltas = compute_ablation_impact(
                sub_peptides, allele, int(pos), method, netmhcpan_path,
                original_scores=sub_original,
            )
            for j, vi in enumerate(valid_idx):
                total_delta[vi] += deltas[j]

    return total_delta


def ablate_random_k(
    peptides: List[str],
    allele: str,
    k: int,
    method: str = "random",
    n_repeats: int = 5,
    netmhcpan_path: Optional[Path] = None,
) -> np.ndarray:
    """Ablate k random positions (averaged over repeats)."""
    valid_lengths = np.array([len(p) for p in peptides])
    # Cache original scores once for the whole peptide set
    all_original = netmhcpan_score_batch(peptides, allele, netmhcpan_path)
    all_deltas = []

    for _ in range(n_repeats):
        total_delta = np.zeros(len(peptides))
        for _ in range(k):
            positions = np.array([
                random.randint(0, max(0, vl - 1)) for vl in valid_lengths
            ])
            for pos in np.unique(positions):
                mask = positions == pos
                idx = np.where(mask)[0]
                sub_peptides = [peptides[i] for i in idx]
                sub_original = all_original[np.array(idx)]
                deltas = compute_ablation_impact(
                    sub_peptides, allele, int(pos), method, netmhcpan_path,
                    original_scores=sub_original,
                )
                for j, vi in enumerate(idx):
                    total_delta[vi] += deltas[j]
        all_deltas.append(total_delta)

    return np.mean(all_deltas, axis=0)


# ============================================================================
# POSITION-LEVEL CORRELATION
# ============================================================================

def compute_shap_delta_correlation(
    peptides: List[str],
    allele: str,
    shap_values: np.ndarray,
    method: str = "controlled",
    max_samples: int = 500,
    netmhcpan_path: Optional[Path] = None,
) -> Tuple[float, float]:
    """
    Compute Spearman correlation between |SHAP| and ablation delta
    across all positions and samples.

    Batches NetMHCpan calls to avoid O(N*L) subprocess invocations:
    original scores are computed once, and per-position mutations for
    each peptide are grouped into a single batched call.
    """
    N = min(len(peptides), max_samples)
    indices = (
        np.random.choice(len(peptides), N, replace=False)
        if len(peptides) > N
        else np.arange(N)
    )

    # Pre-compute original scores for all selected peptides (1 call)
    subset_peps = [peptides[i] for i in indices]
    original_scores = netmhcpan_score_batch(subset_peps, allele, netmhcpan_path)

    # Generate ALL per-position mutations and batch them
    all_mutated: List[str] = []
    mutation_meta: List[Tuple[int, int]] = []  # (local_idx, pos)
    for local_idx, global_idx in enumerate(indices):
        pep = peptides[global_idx]
        for pos in range(len(pep)):
            if method == "controlled":
                all_mutated.append(mutate_controlled(pep, pos))
            elif method == "alanine":
                all_mutated.append(mutate_alanine(pep, pos))
            elif method == "background":
                all_mutated.append(mutate_background(pep, pos))
            else:
                candidates = [aa for aa in STANDARD_AAS if aa != pep[pos]]
                new_aa = random.choice(candidates) if candidates else "A"
                all_mutated.append(pep[:pos] + new_aa + pep[pos + 1:])
            mutation_meta.append((local_idx, pos))

    # Score all mutations in one batched call
    all_ablated = netmhcpan_score_batch(all_mutated, allele, netmhcpan_path)

    # Unpack into (|SHAP|, Δ) pairs
    all_shap_vals: List[float] = []
    all_deltas: List[float] = []
    for k, (local_idx, pos) in enumerate(mutation_meta):
        global_idx = indices[local_idx]
        pep = peptides[global_idx]
        aa_idx = AA_TO_IDX.get(pep[pos], 0)
        shap_mag = abs(float(shap_values[global_idx, pos, aa_idx]))
        delta = float(original_scores[local_idx] - all_ablated[k])
        all_shap_vals.append(shap_mag)
        all_deltas.append(delta)

    if len(all_shap_vals) < 10:
        return 0.0, 1.0

    _sres: Any = np.asarray(stats.spearmanr(all_shap_vals, all_deltas))
    return float(_sres[0]), float(_sres[1])


# ============================================================================
# PIPELINE ABLATION (FULL VS NO-SHAP)
# ============================================================================

def run_pipeline_ablation(
    full_run: Optional[Path] = None,
    ablation_run: Optional[Path] = None,
    alleles: str = "all",
    out: Path = Path("ablation/pipeline_ablation_results.json"),
) -> Dict[str, Any]:
    """
    Evaluate end-to-end SHAP contribution in the generation pipeline.

    Full condition:
        generator -> predictor scoring -> SHAP-guided refinement
        -> predictor rescoring -> selection

    Ablation condition:
        generator -> predictor scoring -> selection (no SHAP refinement)
    """
    from evaluation.evaluate import (
        evaluate_condition,
        find_latest_run,
        load_condition_results,
        print_ablation_delta,
        print_comparison_table,
        resolve_alleles,
    )

    root = ROOT
    full_dir = full_run or find_latest_run(root / "outputs" / "ms_run1")
    abl_dir = ablation_run or find_latest_run(root / "outputs" / "ablation_no_xai")

    print("=" * 70)
    print("PIPELINE ABLATION: EFFECT OF SHAP-GUIDED REFINEMENT")
    print("=" * 70)
    print(f"Full run:     {full_dir}")
    print(f"Ablation run: {abl_dir}")

    alleles_list = resolve_alleles(alleles, full_dir)

    full_eval = evaluate_condition(
        "Full (AR + SHAP refinement)",
        load_condition_results(full_dir, alleles_list),
    )
    abl_eval = evaluate_condition(
        "Ablation (no SHAP refinement)",
        load_condition_results(abl_dir, alleles_list),
    )

    print_comparison_table(
        [full_eval, abl_eval],
        "Ablation: Full Pipeline vs No-SHAP Pipeline",
    )
    print_ablation_delta(full_eval, abl_eval)

    full_macro = full_eval.get("macro_avg", {})
    abl_macro = abl_eval.get("macro_avg", {})
    metric_keys = sorted(set(full_macro) | set(abl_macro))
    delta_macro = {
        key: float(full_macro.get(key, 0.0) - abl_macro.get(key, 0.0))
        for key in metric_keys
    }

    payload: Dict[str, Any] = {
        "mode": "pipeline",
        "definition": {
            "full": "generator -> predictor scoring -> SHAP-guided refinement -> predictor rescoring -> selection",
            "ablation": "generator -> predictor scoring -> selection (no SHAP step)",
        },
        "run_paths": {
            "full_run": str(full_dir),
            "ablation_run": str(abl_dir),
        },
        "alleles": alleles_list,
        "conditions": {
            full_eval["label"]: full_eval,
            abl_eval["label"]: abl_eval,
        },
        "delta_macro_full_minus_ablation": delta_macro,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✅ Pipeline ablation results saved to {out}")
    return payload


# ============================================================================
# V1 PIPELINE
# ============================================================================

def run_ablation_testing(
    shap_json_path: str = "explainer/shap_results.json",
    shap_npz_path: str = "explainer/shap_heatmaps.npz",
    data_path: str = "data/mhc_ab.npz",
    n_samples: int = N_SAMPLES,
    netmhcpan_path: Optional[str] = None,
):
    """
    V1 SHAP ablation analysis: top-k vs bottom-k vs random-k using SHAP
    rankings. Operates directly on NetMHCpan with no surrogate model.
    """
    print("=" * 70)
    print("SHAP ABLATION ANALYSIS (V1)")
    print("=" * 70)

    # Normalize path types (CLI passes str, internal API expects Path | None)
    netmhcpan_p: Optional[Path] = Path(netmhcpan_path) if netmhcpan_path else None

    print("\n[1/2] Loading SHAP results and data...")
    with open(shap_json_path, encoding="utf-8") as f:
        shap_results = json.load(f)
    shap_data = np.load(shap_npz_path, allow_pickle=True)
    raw_data = np.load(data_path, allow_pickle=True)

    all_alleles_raw = np.array([str(a) for a in raw_data["allele"]])
    all_peptides_raw = np.array([str(p) for p in raw_data["peptide"]])

    results = {}

    print("\n[2/2] Running ablation tests...")
    for allele in shap_results:
        print(f"\n{'='*60}")
        print(f"Allele: {allele}")
        print(f"{'='*60}")

        allele_key = allele.replace("*", "_").replace(":", "_")
        allele_result = {"top_k": {}, "bottom_k": {}, "random_k": {}}

        for pep_len in PEP_LENGTHS:
            npz_key = f"{allele_key}_{pep_len}"
            if npz_key not in shap_data:
                continue

            shap_vals = shap_data[npz_key]
            N_shap = shap_vals.shape[0]

            # Load aligned foreground peptides saved by xai_pipeline
            pep_key = f"{allele_key}_{pep_len}_peptides"
            if pep_key in shap_data:
                fg_peptides = shap_data[pep_key].tolist()
                if len(fg_peptides) != N_shap:
                    print("    \u26a0 peptide/SHAP length mismatch, skipping")
                    continue
            else:
                # Legacy NPZ without peptide lists — fall back to pool order
                print(f"    ⚠ No peptide list in NPZ for {npz_key}, using pool fallback")
                mask = (all_alleles_raw == allele) & np.array(
                    [len(p) == pep_len for p in all_peptides_raw]
                )
                fg_peptides = all_peptides_raw[mask].tolist()[:N_shap]

            sample_n = min(n_samples, N_shap)
            if sample_n < 10:
                continue

            idx = np.random.choice(N_shap, sample_n, replace=False)
            peptides = [fg_peptides[i] for i in idx]
            shap_sub = shap_vals[idx]

            print(f"\n  [{pep_len}-mers] testing {len(peptides)} samples")

            for k in K_VALUES:
                delta_top = ablate_top_k(peptides, allele, shap_sub, k, "random", netmhcpan_p)
                delta_bot = ablate_bottom_k(peptides, allele, shap_sub, k, "random", netmhcpan_p)
                delta_rand = ablate_random_k(peptides, allele, k, "random", 3, netmhcpan_p)

                # Statistical significance (paired Wilcoxon signed-rank)
                try:
                    _wres: Any = stats.wilcoxon(
                        delta_top, delta_bot, alternative="greater"
                    )
                    wilcox_p = float(_wres.pvalue)
                except ValueError:
                    wilcox_p = 1.0

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (np.var(delta_top) + np.var(delta_bot)) / 2
                ) + 1e-9
                cohens_d = (np.mean(delta_top) - np.mean(delta_bot)) / pooled_std

                key = f"{pep_len}mer_k{k}"
                allele_result["top_k"][key] = {
                    "mean": float(np.mean(delta_top)),
                    "std": float(np.std(delta_top)),
                }
                allele_result["bottom_k"][key] = {
                    "mean": float(np.mean(delta_bot)),
                    "std": float(np.std(delta_bot)),
                }
                allele_result["random_k"][key] = {
                    "mean": float(np.mean(delta_rand)),
                    "std": float(np.std(delta_rand)),
                }
                allele_result.setdefault("stats", {})[key] = {
                    "wilcoxon_p": float(wilcox_p),
                    "cohens_d": float(cohens_d),
                    "top_gt_random": bool(
                        np.mean(delta_top) > np.mean(delta_rand)
                    ),
                }

                ratio = np.mean(delta_top) / (np.mean(delta_bot) + 1e-6)
                sig = "p<0.05" if wilcox_p < 0.05 else f"p={wilcox_p:.3f}"
                is_pass = (
                    np.mean(delta_top) > np.mean(delta_bot) and wilcox_p < 0.05
                )
                status = "PASS ✓" if is_pass else "FAIL ✗"
                print(
                    f"    k={k}: Top-k Δ={np.mean(delta_top):.4f}  "
                    f"Bot-k Δ={np.mean(delta_bot):.4f}  "
                    f"Rand-k Δ={np.mean(delta_rand):.4f}  "
                    f"Ratio={ratio:.2f}x  d={cohens_d:.2f}  {sig}  {status}"
                )

        results[allele] = allele_result

    # SHAP ablation summary
    print("\n" + "=" * 70)
    print("SHAP ABLATION SUMMARY (V1)")
    print("=" * 70)
    n_pass, n_total = 0, 0
    for allele, ar in results.items():
        for key in ar.get("stats", {}):
            n_total += 1
            s = ar["stats"][key]
            if (
                ar["top_k"][key]["mean"] > ar["bottom_k"].get(key, {}).get("mean", 0)
                and s["wilcoxon_p"] < 0.05
            ):
                n_pass += 1
    all_pass = n_pass == n_total and n_total > 0
    print("  Statistically significant (p<0.05) top-k > bottom-k:")
    print(f"    {n_pass}/{n_total} buckets passed")
    print(f"  Overall: {'PASS ✅' if all_pass else 'FAIL ❌'}")

    output = {"results": results, "faithfulness_all_pass": all_pass}
    out_path = Path("ablation/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")
    return output


# ============================================================================
# V2 (ENHANCED) PIPELINE
# ============================================================================

def run_enhanced_analysis(
    shap_json_path: str = "explainer/shap_results.json",
    shap_npz_path: str = "explainer/shap_heatmaps.npz",
    data_path: str = "data/mhc_ab.npz",
    n_samples: int = N_SAMPLES,
    netmhcpan_path: Optional[str] = None,
):
    """
    Enhanced SHAP ablation analysis:
      A) Controlled mutation ablation (top-k vs bottom-k)
      B) Stratified analysis by binding strength
      C) |SHAP| vs Δ correlation
    """
    print("=" * 70)
    print("ENHANCED SHAP ABLATION ANALYSIS (V2)")
    print("=" * 70)

    # Normalize path types (CLI passes str, internal API expects Path | None)
    netmhcpan_p: Optional[Path] = Path(netmhcpan_path) if netmhcpan_path else None

    with open(shap_json_path, encoding="utf-8") as f:
        shap_results = json.load(f)
    shap_data = np.load(shap_npz_path, allow_pickle=True)
    raw_data = np.load(data_path, allow_pickle=True)

    all_alleles_raw = np.array([str(a) for a in raw_data["allele"]])
    all_peptides_raw = np.array([str(p) for p in raw_data["peptide"]])

    all_output = {}

    for allele in shap_results:
        print(f"\n{'='*60}")
        print(f"Allele: {allele}")
        print(f"{'='*60}")

        allele_key = allele.replace("*", "_").replace(":", "_")
        allele_out: dict = {
            "controlled_ablation": {},
            "stratified": {},
            "correlation": {},
        }

        for pep_len in PEP_LENGTHS:
            npz_key = f"{allele_key}_{pep_len}"
            if npz_key not in shap_data:
                continue

            shap_vals = shap_data[npz_key]
            N_shap = shap_vals.shape[0]

            # Load aligned foreground peptides saved by xai_pipeline
            pep_key = f"{allele_key}_{pep_len}_peptides"
            if pep_key in shap_data:
                fg_peptides = shap_data[pep_key].tolist()
                if len(fg_peptides) != N_shap:
                    print("    \u26a0 peptide/SHAP length mismatch, skipping")
                    continue
            else:
                # Legacy NPZ without peptide lists — fall back to pool order
                print(f"    ⚠ No peptide list in NPZ for {npz_key}, using pool fallback")
                mask = (all_alleles_raw == allele) & np.array(
                    [len(p) == pep_len for p in all_peptides_raw]
                )
                fg_peptides = all_peptides_raw[mask].tolist()[:N_shap]

            sample_n = min(n_samples, N_shap)
            if sample_n < 10:
                continue

            idx = np.random.choice(N_shap, sample_n, replace=False)
            peptides = [fg_peptides[i] for i in idx]
            shap_sub = shap_vals[idx]

            bucket = f"{pep_len}mer"
            print(f"\n  [{bucket}] n={len(peptides)}")

            # A) Controlled mutation top-k vs bottom-k
            print("    A) Controlled top-k vs bottom-k...")
            ctrl_result = {}
            for k in K_VALUES:
                dt = ablate_top_k(peptides, allele, shap_sub, k, "controlled", netmhcpan_p)
                db = ablate_bottom_k(peptides, allele, shap_sub, k, "controlled", netmhcpan_p)
                try:
                    _wres: Any = stats.wilcoxon(dt, db, alternative="greater")
                    wilcox_p = float(_wres.pvalue)
                except ValueError:
                    wilcox_p = 1.0
                pooled_std = np.sqrt((np.var(dt) + np.var(db)) / 2) + 1e-9
                cohens_d = (np.mean(dt) - np.mean(db)) / pooled_std
                ctrl_result[f"k{k}"] = {
                    "top_k_mean": float(np.mean(dt)),
                    "bottom_k_mean": float(np.mean(db)),
                    "ratio": float(np.mean(dt) / (np.mean(db) + 1e-6)),
                    "wilcoxon_p": float(wilcox_p),
                    "cohens_d": float(cohens_d),
                }
                sig = "p<0.05" if wilcox_p < 0.05 else f"p={wilcox_p:.3f}"
                is_pass = np.mean(dt) > np.mean(db) and wilcox_p < 0.05
                status = "✓" if is_pass else "✗"
                print(f"       k={k}: Top={np.mean(dt):.4f} Bot={np.mean(db):.4f} d={cohens_d:.2f} {sig} {status}")
            allele_out["controlled_ablation"][bucket] = ctrl_result

            # B) Stratified by binding strength
            print("    B) Stratified analysis...")
            scores = netmhcpan_score_batch(peptides, allele, netmhcpan_p)
            p25, p75 = np.percentile(scores, [25, 75])
            strong_idx = np.where(scores >= p75)[0]
            weak_idx = np.where(scores <= p25)[0]

            strat_result = {}
            for name, sub_idx in [("strong", strong_idx), ("weak", weak_idx)]:
                if len(sub_idx) < 10:
                    continue
                sub_peps = [peptides[i] for i in sub_idx]
                sub_shap = shap_sub[sub_idx]
                dt = ablate_top_k(sub_peps, allele, sub_shap, 2, "controlled", netmhcpan_p)
                db = ablate_bottom_k(sub_peps, allele, sub_shap, 2, "controlled", netmhcpan_p)
                strat_result[name] = {
                    "n": len(sub_idx),
                    "top_k_mean": float(np.mean(dt)),
                    "bottom_k_mean": float(np.mean(db)),
                    "ratio": float(np.mean(dt) / (np.mean(db) + 1e-6)),
                }
                print(f"       {name}: Top={np.mean(dt):.4f} Bot={np.mean(db):.4f}")
            allele_out["stratified"][bucket] = strat_result

            # C) Correlation
            print("    C) SHAP-Δ correlation...")
            corr_n = min(200, len(peptides))
            r, p = compute_shap_delta_correlation(
                peptides[:corr_n], allele, shap_sub[:corr_n],
                "controlled", corr_n, netmhcpan_p
            )
            allele_out["correlation"][bucket] = {
                "spearman_r": float(r), "p_value": float(p), "n": corr_n
            }
            sig = "✓" if p < 0.05 else "✗"
            print(f"       r={r:.4f}, p={p:.2e} {sig}")

        # Compute an overall SHAP ablation score from the main analysis signals.
        mono_passes = 0
        mono_total = 0
        for bucket, ctrl in allele_out["controlled_ablation"].items():
            for _, kval in ctrl.items():
                mono_total += 1
                if (kval["top_k_mean"] > kval["bottom_k_mean"]
                        and kval.get("wilcoxon_p", 1.0) < 0.05):
                    mono_passes += 1

        monotonicity = mono_passes / max(mono_total, 1)

        corr_scores = [v["spearman_r"] for v in allele_out["correlation"].values()]
        avg_corr = float(np.mean(corr_scores)) if corr_scores else 0.0

        faithfulness = 0.6 * monotonicity + 0.4 * max(0, min(1, avg_corr / 0.2))
        allele_out["faithfulness_score"] = float(faithfulness)
        allele_out["faithfulness_components"] = {
            "monotonicity": float(monotonicity),
            "avg_correlation": float(avg_corr),
        }

        print(f"\n  SHAP ablation score: {faithfulness:.3f}")
        print(f"    Monotonicity: {monotonicity:.3f}  Correlation: {avg_corr:.4f}")

        all_output[allele] = allele_out

    out_path = Path("ablation/enhanced_ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_output, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")
    return all_output


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Pipeline ablation (full vs no-SHAP) with optional "
        "SHAP ablation analysis modes"
    ),
)
@click.option(
    "--full_run",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to full pipeline run directory",
)
@click.option(
    "--ablation_run",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to no-SHAP ablation run directory",
)
@click.option(
    "--alleles",
    default="all",
    show_default=True,
    help='Comma-separated alleles or "all" (resolved from full run summary)',
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("ablation/pipeline_ablation_results.json"),
    show_default=True,
    help="Output path for pipeline ablation mode",
)
@click.option(
    "--data",
    type=click.Path(path_type=Path),
    default=Path("data/mhc_ab.npz"),
    show_default=True,
    help="Path to NPZ data",
)
@click.option(
    "--shap_json",
    type=click.Path(path_type=Path),
    default=Path("explainer/shap_results.json"),
    show_default=True,
    help="Path to SHAP results JSON",
)
@click.option(
    "--shap_npz",
    type=click.Path(path_type=Path),
    default=Path("explainer/shap_heatmaps.npz"),
    show_default=True,
    help="Path to SHAP heatmaps NPZ",
)
@click.option(
    "--netmhcpan",
    type=click.Path(path_type=Path),
    default=None,
    help="NetMHCpan install path",
)
@click.option(
    "--mode",
    type=click.Choice(["pipeline", "v1", "enhanced"], case_sensitive=True),
    default="pipeline",
    show_default=True,
    help="pipeline = full vs no-SHAP, v1/enhanced = SHAP ablation analyses",
)
@click.option(
    "--n_samples",
    type=int,
    default=N_SAMPLES,
    show_default=True,
    help="Samples per allele",
)
def main(
    full_run: Path | None,
    ablation_run: Path | None,
    alleles: str,
    out: Path,
    data: Path,
    shap_json: Path,
    shap_npz: Path,
    netmhcpan: Path | None,
    mode: str,
    n_samples: int,
) -> None:
    """Run pipeline ablation or SHAP ablation analyses."""
    if mode == "pipeline":
        log_path = out.with_suffix(".log")
    elif mode == "v1":
        log_path = Path("ablation/ablation_results.log")
    else:
        log_path = Path("ablation/enhanced_ablation_results.log")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            try:
                if mode == "pipeline":
                    run_pipeline_ablation(
                        full_run=full_run,
                        ablation_run=ablation_run,
                        alleles=alleles,
                        out=out,
                    )
                    return

                if mode == "v1":
                    run_ablation_testing(
                        shap_json_path=str(shap_json),
                        shap_npz_path=str(shap_npz),
                        data_path=str(data),
                        n_samples=n_samples,
                        netmhcpan_path=str(netmhcpan) if netmhcpan else None,
                    )
                    return

                run_enhanced_analysis(
                    shap_json_path=str(shap_json),
                    shap_npz_path=str(shap_npz),
                    data_path=str(data),
                    n_samples=n_samples,
                    netmhcpan_path=str(netmhcpan) if netmhcpan else None,
                )
            except Exception:
                traceback.print_exc()
                raise SystemExit(1) from None


if __name__ == "__main__":
    main()
