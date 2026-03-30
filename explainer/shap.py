#!/usr/bin/env python3
"""
Direct NetMHCpan SHAP Pipeline for HLA-Peptide Binding

Architecture:
    SHAP is run directly against NetMHCpan 4.2 (no surrogate model).
    For each (allele, peptide-length) bucket, peptides are encoded as
    categorical position indices and explained with KernelSHAP.
    This keeps explanations tied to the real predictor.

Workflow:
    1. For each allele + length (8, 9, 10, 11):
       a. Build background and foreground sets
       b. Run KernelSHAP on direct NetMHCpan scores
       c. Aggregate SHAP values → position x amino-acid heatmap
    2. Merge heatmaps across lengths into a single per-allele profile
    3. Output JSON + NPZ consumed by the generator's refinement module

Outputs:
    explainer/shap_results.json
        per-allele position weights, heatmaps, metadata
    explainer/shap_heatmaps.npz
        raw SHAP heatmaps (allelexlength→(N, L, 20))
"""

import importlib
import json
import os
import platform
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import traceback

import click
import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AAS)}
IDX_TO_AA = dict(enumerate(STANDARD_AAS))
NUM_AAS = len(STANDARD_AAS)  # 20

PEP_LENGTHS = [8, 9, 10, 11]
DEFAULT_BACKGROUND_SIZE = 64
DEFAULT_FOREGROUND_SIZE = 128
DEFAULT_NSAMPLES = 32  # Lower default KernelSHAP budget for faster runs.

# ============================================================================
# NETMHCPAN WRAPPER
# ============================================================================


def _resolve_netmhcpan(
    netmhcpan_path: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve path to NetMHCpan 4.2 binary.

    Returns:
        (binary_path, home_directory)
    """
    if netmhcpan_path and Path(netmhcpan_path).exists():
        home = Path(netmhcpan_path)
    elif os.environ.get("NETMHCPAN"):
        home = Path(os.environ["NETMHCPAN"])
    else:
        # Sibling directory relative to project root
        project_root = Path(__file__).resolve().parent.parent
        home = project_root.parent / "netMHCpan-4.2"
    if not home.exists():
        raise FileNotFoundError(
            f"NetMHCpan not found at {home}. "
            "Set NETMHCPAN env var or pass --netmhcpan."
        )
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        binary = home / "Darwin_arm64" / "bin" / "netMHCpan-4.2"
    elif system == "Darwin":
        binary = home / "Darwin_x86_64" / "bin" / "netMHCpan-4.2"
    else:
        binary = home / "Linux_x86_64" / "bin" / "netMHCpan-4.2"
    if not binary.exists():
        raise FileNotFoundError(f"NetMHCpan binary not found at {binary}")
    return binary, home


# pylint: disable=too-many-locals
def run_netmhcpan(
    peptides: list[str],
    allele: str,
    netmhcpan_path: Path | None = None,
    timeout: int = 120,
) -> dict[str, float]:
    """
    Run NetMHCpan 4.2 on a list of peptides for one allele.
    """
    binary, home = _resolve_netmhcpan(netmhcpan_path)
    netmhc_allele = allele.replace("*", "")

    # Write peptides to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as handle:
        for pep in peptides:
            handle.write(f"{pep}\n")
        tmp_path = Path(handle.name)

    env = os.environ.copy()
    env["NMHOME"] = str(home)
    env["NETMHCpan"] = str(home)
    env["TMPDIR"] = tempfile.gettempdir()

    try:
        result = subprocess.run(
            [str(binary), "-p", str(tmp_path), "-a", netmhc_allele, "-BA"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(home),
            check=False,
        )
        affinities: dict[str, float] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 16 and parts[0].isdigit():
                try:
                    pep = parts[2]
                    aff_nm = float(parts[15])
                    affinities[pep] = aff_nm
                except ValueError:
                    continue
        return affinities
    except subprocess.TimeoutExpired:
        return {}
    finally:
        tmp_path.unlink(missing_ok=True)


# pylint: enable=too-many-locals


def netmhcpan_score_batch(
    peptides: list[str],
    allele: str,
    netmhcpan_path: Path | None = None,
    batch_size: int = 2000,
    netmhcpan_jobs: int | None = None,
) -> np.ndarray:
    """
    Score peptides via NetMHCpan and return -log10(nM) scores.
    Higher = better binder (matches generator convention).
    """
    all_scores = np.zeros(len(peptides))
    jobs = netmhcpan_jobs
    if jobs is None:
        jobs = int(os.environ.get("NETMHCPAN_JOBS", "1"))
    jobs = max(1, jobs)

    batches = [
        (start, peptides[start:start + batch_size])
        for start in range(0, len(peptides), batch_size)
    ]
    if jobs <= 1 or len(batches) <= 1:
        for start, batch in batches:
            aff_map = run_netmhcpan(batch, allele, netmhcpan_path)
            for i, pep in enumerate(batch):
                aff = aff_map.get(pep, 50000.0)  # default = very weak binder
                all_scores[start + i] = -np.log10(max(aff, 1e-9))
        return all_scores

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_map = {
            executor.submit(run_netmhcpan, batch, allele, netmhcpan_path): (
                start,
                batch,
            )
            for start, batch in batches
        }
        for future in as_completed(future_map):
            start, batch = future_map[future]
            try:
                aff_map = future.result()
            except Exception:
                aff_map = {}
            for i, pep in enumerate(batch):
                aff = aff_map.get(pep, 50000.0)
                all_scores[start + i] = -np.log10(max(aff, 1e-9))
    return all_scores


# ============================================================================
# PEPTIDE ENCODING FOR SHAP
# ============================================================================

def encode_peptide_positions(peptide: str) -> np.ndarray:
    """Encode a peptide as integer amino-acid indices (shape: (L,))."""
    vec = np.zeros(len(peptide), dtype=np.float64)
    for i, aa_char in enumerate(peptide):
        vec[i] = float(AA_TO_IDX.get(aa_char, 0))
    return vec


def decode_position_vector(vec: np.ndarray) -> str:
    """Decode a position-index vector to a peptide string."""
    idxs = np.rint(np.asarray(vec)).astype(int)
    idxs = np.clip(idxs, 0, NUM_AAS - 1)
    return "".join(IDX_TO_AA[i] for i in idxs.tolist())


def encode_peptides_position_matrix(peptides: list[str]) -> np.ndarray:
    """Encode same-length peptides into matrix shape (N, L)."""
    if not peptides:
        return np.empty((0, 0), dtype=np.float64)
    mat = np.zeros((len(peptides), len(peptides[0])), dtype=np.float64)
    for i, pep in enumerate(peptides):
        mat[i] = encode_peptide_positions(pep)
    return mat


def _score_position_matrix_with_netmhcpan(
    matrix: np.ndarray,
    allele: str,
    pep_length: int,
    netmhcpan_path: Path | None = None,
    netmhcpan_jobs: int | None = None,
) -> np.ndarray:
    """
    Model callback used by KernelSHAP.
    Converts position-index vectors into peptides and scores them with
    NetMHCpan.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < pep_length:
        raise ValueError(
            f"Expected at least {pep_length} features, got {arr.shape[1]}"
        )
    if arr.shape[1] > pep_length:
        arr = arr[:, :pep_length]

    peptides = [decode_position_vector(row) for row in arr]
    unique_peptides = list(dict.fromkeys(peptides))
    unique_scores = netmhcpan_score_batch(
        unique_peptides,
        allele,
        netmhcpan_path,
        netmhcpan_jobs=netmhcpan_jobs,
    )
    score_map = {
        pep: float(score) for pep, score in zip(unique_peptides, unique_scores)
    }
    return np.array([score_map[p] for p in peptides], dtype=np.float64)


# pylint: disable=too-many-locals
def run_direct_netmhcpan_shap_bucket(
    allele: str,
    pep_length: int,
    background_peptides: list[str],
    foreground_peptides: list[str],
    *,
    netmhcpan_path: Path | None = None,
    nsamples: int = DEFAULT_NSAMPLES,
    netmhcpan_jobs: int | None = None,
) -> np.ndarray:  # pylint: disable=too-many-arguments
    """
    Run KernelSHAP directly against NetMHCpan scores for one bucket.
    """
    shap = importlib.import_module("shap")

    bg_matrix = encode_peptides_position_matrix(background_peptides)
    fg_matrix = encode_peptides_position_matrix(foreground_peptides)

    def _model_fn(position_matrix: np.ndarray) -> np.ndarray:
        return _score_position_matrix_with_netmhcpan(
            position_matrix,
            allele=allele,
            pep_length=pep_length,
            netmhcpan_path=netmhcpan_path,
            netmhcpan_jobs=netmhcpan_jobs,
        )

    explainer = shap.KernelExplainer(_model_fn, bg_matrix)
    raw_shap = explainer.shap_values(fg_matrix, nsamples=nsamples)
    if isinstance(raw_shap, list):
        raw_shap = raw_shap[0]
    raw_shap = np.asarray(raw_shap, dtype=np.float64)
    if raw_shap.ndim == 1:
        raw_shap = raw_shap.reshape(1, -1)

    # KernelSHAP here returns per-position attributions (N, L).
    # Expand to (N, L, 20), placing attribution at the actually observed AA.
    num_foreground = len(foreground_peptides)
    shap_3d = np.zeros((num_foreground, pep_length, NUM_AAS), dtype=np.float64)
    for i, pep in enumerate(foreground_peptides):
        for pos, aa_char in enumerate(pep):
            aa_idx = AA_TO_IDX.get(aa_char)
            if aa_idx is not None and pos < raw_shap.shape[1]:
                shap_3d[i, pos, aa_idx] = raw_shap[i, pos]

    return shap_3d


# pylint: enable=too-many-locals


def build_background_set(
    peptides: list[str],
    allele: str,
    pep_length: int,
    n_background: int = DEFAULT_BACKGROUND_SIZE,
    netmhcpan_path: Path | None = None,
    netmhcpan_jobs: int | None = None,
) -> list[str]:
    """
    Build a background set of 'typical' peptides for one allele/length bucket.
    Samples uniformly from the available peptides of that length,
    biased toward mid-range binders.
    """
    pool = [p for p in peptides if len(p) == pep_length]
    if len(pool) <= n_background:
        return pool

    # Score all candidates
    scores = netmhcpan_score_batch(
        pool,
        allele,
        netmhcpan_path,
        netmhcpan_jobs=netmhcpan_jobs,
    )

    # Pick mid-range binders (25th-75th percentile) to be representative
    q25, q75 = np.percentile(scores, [25, 75])
    mid_mask = (scores >= q25) & (scores <= q75)
    mid_indices = np.where(mid_mask)[0]

    if len(mid_indices) >= n_background:
        chosen = np.random.choice(mid_indices, n_background, replace=False)
    else:
        # Fill with mid first, then random from remainder
        remaining = np.setdiff1d(np.arange(len(pool)), mid_indices)
        extra = np.random.choice(
            remaining,
            min(n_background - len(mid_indices), len(remaining)),
            replace=False,
        )
        chosen = np.concatenate([mid_indices, extra])

    return [pool[i] for i in chosen]


# pylint: disable=too-many-locals
def select_foreground_set(
    peptides: list[str],
    allele: str,
    pep_length: int,
    n_foreground: int = DEFAULT_FOREGROUND_SIZE,
    netmhcpan_path: Path | None = None,
    netmhcpan_jobs: int | None = None,
) -> list[str]:
    """
    Select foreground peptides to explain.
    This ensures SHAP captures both strong-binding motifs and general trends.
    """
    pool = [p for p in peptides if len(p) == pep_length]
    if len(pool) <= n_foreground:
        return pool

    scores = netmhcpan_score_batch(
        pool,
        allele,
        netmhcpan_path,
        netmhcpan_jobs=netmhcpan_jobs,
    )

    # 60% from top quartile, 40% from mid range
    n_top = int(n_foreground * 0.6)
    n_mid = n_foreground - n_top

    q50, q75 = np.percentile(scores, [50, 75])
    top_idx = np.where(scores >= q75)[0]
    mid_idx = np.where((scores >= q50) & (scores < q75))[0]

    chosen_top = np.random.choice(
        top_idx, min(n_top, len(top_idx)), replace=False
    )
    chosen_mid = np.random.choice(
        mid_idx, min(n_mid, len(mid_idx)), replace=False
    )
    chosen = np.concatenate([chosen_top, chosen_mid])

    # If we don't have enough, fill from remainder
    if len(chosen) < n_foreground:
        remaining = np.setdiff1d(np.arange(len(pool)), chosen)
        extra = np.random.choice(
            remaining,
            min(n_foreground - len(chosen), len(remaining)),
            replace=False,
        )
        chosen = np.concatenate([chosen, extra])

    return [pool[int(i)] for i in chosen]


# pylint: enable=too-many-locals


def aggregate_shap_heatmap(
    shap_values: np.ndarray,
    foreground_peptides: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-sample SHAP into a position × AA heatmap.

    Args:
        shap_values:        (N, L, 20) SHAP values
        foreground_peptides: list of N peptides (all same length L)

    Returns:
        heatmap:          (L, 20) mean SHAP per position × AA
        position_weights: (L,) mean |SHAP| per position (for refinement)
    """
    _, num_positions, num_aas = shap_values.shape

    # Weighted average: for each (pos, aa), average SHAP across samples
    # that actually have that AA at that position
    heatmap = np.zeros((num_positions, num_aas))
    counts = np.zeros((num_positions, num_aas))

    for i, pep in enumerate(foreground_peptides):
        for pos, aa_char in enumerate(pep):
            aa_idx = AA_TO_IDX.get(aa_char)
            if aa_idx is not None:
                heatmap[pos, aa_idx] += shap_values[i, pos, aa_idx]
                counts[pos, aa_idx] += 1

    # Avoid division by zero
    safe_counts = np.maximum(counts, 1)
    heatmap = heatmap / safe_counts

    # Position importance = mean |SHAP| across all AAs at each position
    position_weights = np.mean(np.abs(shap_values), axis=(0, 2))  # (L,)

    return heatmap, position_weights


def merge_position_weights(
    per_length_weights: dict[int, np.ndarray],
    max_positions: int = 11,
) -> np.ndarray:
    """
    Merge per-length position weights into a single (max_positions,) array
    using N/C alignment (P1, P2, … from N-term; PΩ from C-term).
    """
    merged = np.zeros(max_positions)
    weight_counts = np.zeros(max_positions)

    for length, weights in per_length_weights.items():
        if len(weights) == 0:
            continue

        # N-terminal positions: first half
        n_half = (length + 1) // 2
        for i in range(min(n_half, max_positions)):
            merged[i] += weights[i]
            weight_counts[i] += 1

        # C-terminal positions: align from the end
        for c_offset in range(length - n_half):
            src_idx = length - 1 - c_offset
            dst_idx = max_positions - 1 - c_offset
            if dst_idx >= 0 and src_idx < len(weights):
                merged[dst_idx] += weights[src_idx]
                weight_counts[dst_idx] += 1

    safe_counts = np.maximum(weight_counts, 1)
    merged = merged / safe_counts

    # Normalise to sum to 1
    total = merged.sum()
    if total > 0:
        merged = merged / total

    return merged


def extract_anchor_motifs(
    heatmaps: dict[int, np.ndarray],
    top_n: int = 5,
) -> dict[str, list[dict[str, float | str]]]:
    """
    Extract top amino acids at P2 and PΩ from heatmaps.
    """
    p2_scores: defaultdict[str, float] = defaultdict(float)
    p2_counts: defaultdict[str, int] = defaultdict(int)
    pomega_scores: defaultdict[str, float] = defaultdict(float)
    pomega_counts: defaultdict[str, int] = defaultdict(int)

    for _, heatmap in heatmaps.items():
        if heatmap.shape[0] < 2:
            continue
        # P2 = index 1
        for aa_idx in range(NUM_AAS):
            aa_char = STANDARD_AAS[aa_idx]
            p2_scores[aa_char] += heatmap[1, aa_idx]
            p2_counts[aa_char] += 1
        # PΩ = last position
        for aa_idx in range(NUM_AAS):
            aa_char = STANDARD_AAS[aa_idx]
            pomega_scores[aa_char] += heatmap[-1, aa_idx]
            pomega_counts[aa_char] += 1

    # Average
    p2_avg = {
        aa: p2_scores[aa] / max(p2_counts[aa], 1) for aa in STANDARD_AAS
    }
    pomega_avg = {
        aa: pomega_scores[aa] / max(pomega_counts[aa], 1)
        for aa in STANDARD_AAS
    }

    p2_ranked = sorted(
        p2_avg.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    pomega_ranked = sorted(
        pomega_avg.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    return {
        "P2": [{"aa": aa, "shap_score": float(s)} for aa, s in p2_ranked],
        "P_omega": [
            {"aa": aa, "shap_score": float(s)}
            for aa, s in pomega_ranked
        ],
    }


# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def run_shap_pipeline(
    *,
    data_path: str = "data/mhc_ab.npz",
    netmhcpan_path: str | None = None,
    n_background: int = DEFAULT_BACKGROUND_SIZE,
    n_foreground: int = DEFAULT_FOREGROUND_SIZE,
    nsamples: int = DEFAULT_NSAMPLES,
    netmhcpan_jobs: int | None = None,
    alleles: list[str] | None = None,
    max_alleles: int = 10,
    output_dir: str = "explainer",
) -> dict[str, dict[str, object]]:
    """Run the direct NetMHCpan SHAP pipeline."""
    print("=" * 70)
    print("SHAP PIPELINE — Direct NetMHCpan + KernelSHAP")
    print("=" * 70)

    # Normalise path type early so all downstream calls receive Path | None.
    netmhcpan: Path | None = Path(netmhcpan_path) if netmhcpan_path else None
    _resolve_netmhcpan(netmhcpan)
    print("✅ NetMHCpan 4.2 found")

    # Load data
    print("\n[1/4] Loading data...")
    data = np.load(data_path, allow_pickle=True)

    all_alleles = np.array([str(a) for a in data["allele"]])
    all_peptides = np.array([str(p) for p in data["peptide"]])

    # Filter to 8-11 mers with standard AAs only
    valid_mask = np.array(
        [
            8 <= len(p) <= 11 and all(c in AA_TO_IDX for c in p)
            for p in all_peptides
        ]
    )
    all_alleles = all_alleles[valid_mask]
    all_peptides = all_peptides[valid_mask]
    print(f"   {len(all_peptides)} valid peptides (8-11 mers, standard AAs)")
    jobs = int(netmhcpan_jobs or os.environ.get("NETMHCPAN_JOBS", "1"))
    jobs = max(1, jobs)
    print(f"   NetMHCpan parallel jobs: {jobs}")

    # Select alleles
    if alleles is None:
        unique, counts = np.unique(all_alleles, return_counts=True)
        top_idx = np.argsort(-counts)[:max_alleles]
        alleles = unique[top_idx].tolist()
    assert alleles is not None  # guaranteed by the block above
    print(f"   Processing {len(alleles)} alleles: {alleles}")

    # Per-allele results
    all_results = {}
    raw_heatmaps = {}

    for allele in alleles:
        print(f"\n{'='*60}")
        print(f"Allele: {allele}")
        print(f"{'='*60}")

        allele_mask = all_alleles == allele
        allele_peptides = all_peptides[allele_mask].tolist()

        allele_heatmaps: dict[int, np.ndarray] = {}
        allele_weights: dict[int, np.ndarray] = {}
        allele_shap_raw: dict[int, np.ndarray] = {}
        allele_foreground: dict[int, list[str]] = {}
        bucket_meta: dict[int, dict[str, int | str]] = {}

        for pep_len in PEP_LENGTHS:
            pool = [p for p in allele_peptides if len(p) == pep_len]
            if len(pool) < 10:
                print(
                    f"  {pep_len}-mers: skipping (only {len(pool)} peptides)"
                )
                continue

            print(f"\n  [{pep_len}-mers] pool={len(pool)}")

            # Build background
            print(f"    Building background set (n={n_background})...")
            background_set = build_background_set(
                pool,
                allele,
                pep_len,
                n_background,
                netmhcpan,
                netmhcpan_jobs=jobs,
            )
            print(f"    Background: {len(background_set)} peptides")

            # Select foreground
            print(f"    Selecting foreground set (n={n_foreground})...")
            foreground_set = select_foreground_set(
                pool,
                allele,
                pep_len,
                n_foreground,
                netmhcpan,
                netmhcpan_jobs=jobs,
            )
            print(f"    Foreground: {len(foreground_set)} peptides")

            # Run SHAP directly on NetMHCpan
            print(
                "    Running direct KernelSHAP on NetMHCpan "
                f"(nsamples={nsamples})..."
            )
            shap_vals = run_direct_netmhcpan_shap_bucket(
                allele,
                pep_len,
                background_set,
                foreground_set,
                netmhcpan_path=netmhcpan,
                nsamples=nsamples,
                netmhcpan_jobs=jobs,
            )
            print(f"    SHAP values shape: {shap_vals.shape}")

            # Aggregate
            heatmap, pos_weights = aggregate_shap_heatmap(
                shap_vals,
                foreground_set,
            )
            allele_heatmaps[pep_len] = heatmap
            allele_weights[pep_len] = pos_weights
            allele_shap_raw[pep_len] = shap_vals
            allele_foreground[pep_len] = foreground_set
            bucket_meta[pep_len] = {
                "n_background": len(background_set),
                "n_foreground": len(foreground_set),
                "shap_method": "kernel_shap_direct_netmhcpan",
                "xai_method": "kernel_shap_direct_netmhcpan",
                "nsamples": int(nsamples),
                "netmhcpan_jobs": int(jobs),
            }

            # Print top positions
            ranked_pos = np.argsort(-pos_weights)
            print("    Position importance (ranked):")
            for rank, position_index in enumerate(ranked_pos[:5]):
                print(
                    f"      #{rank+1}: "
                    "P"
                    f"{position_index + 1} = "
                    f"{pos_weights[position_index]:.4f}"
                )

        # Merge position weights across lengths
        merged_weights = merge_position_weights(allele_weights)
        print("\n  Merged position weights (11 positions):")
        for i, weight in enumerate(merged_weights):
            print(f"    P{i+1}: {weight:.4f}")

        # Extract anchor motifs
        motifs = extract_anchor_motifs(allele_heatmaps)
        print("\n  Anchor motifs:")
        if motifs["P2"]:
            print(
                f"    P2:  {', '.join(str(m['aa']) for m in motifs['P2'][:3])}"
            )
        if motifs["P_omega"]:
            print(
                "    PΩ:  "
                f"{', '.join(str(m['aa']) for m in motifs['P_omega'][:3])}"
            )

        # Store results
        allele_key = allele.replace("*", "_").replace(":", "_")
        all_results[allele] = {
            "position_weights": merged_weights.tolist(),
            "per_length_weights": {
                str(k): v.tolist() for k, v in allele_weights.items()
            },
            "heatmaps": {
                str(k): v.tolist() for k, v in allele_heatmaps.items()
            },
            "anchor_motifs": motifs,
            "bucket_meta": bucket_meta,
        }

        # Store raw SHAP for NPZ (values + aligned foreground peptide lists)
        for pep_len, shap_values_array in allele_shap_raw.items():
            raw_heatmaps[f"{allele_key}_{pep_len}"] = shap_values_array
            if pep_len in allele_foreground:
                raw_heatmaps[f"{allele_key}_{pep_len}_peptides"] = np.array(
                    allele_foreground[pep_len]
                )

    # ── Save outputs ──────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "shap_results.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\n✅ Results saved to {json_path}")

    npz_path = out / "shap_heatmaps.npz"
    np.savez_compressed(npz_path, **raw_heatmaps)
    print(f"✅ Raw SHAP heatmaps saved to {npz_path}")

    return all_results


def run_xai_pipeline(**kwargs) -> dict[str, dict[str, object]]:
    """Backward-compatible alias for ``run_shap_pipeline``."""
    return run_shap_pipeline(**kwargs)


def main(
    *,
    data: Path,
    netmhcpan: Path | None,
    n_background: int,
    n_foreground: int,
    nsamples: int,
    netmhcpan_jobs: int | None,
    alleles: str | None,
    max_alleles: int,
    output_dir: Path,
) -> None:
    """Run the SHAP pipeline from the command line."""
    allele_list = None
    if alleles:
        allele_list = [allele.strip() for allele in alleles.split(",")]
    log_path = output_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            try:
                run_shap_pipeline(
                    data_path=str(data),
                    netmhcpan_path=str(netmhcpan) if netmhcpan else None,
                    n_background=n_background,
                    n_foreground=n_foreground,
                    nsamples=nsamples,
                    netmhcpan_jobs=netmhcpan_jobs,
                    alleles=allele_list,
                    max_alleles=max_alleles,
                    output_dir=str(output_dir),
                )
            except Exception:
                traceback.print_exc()
                raise SystemExit(1) from None


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Direct NetMHCpan + KernelSHAP SHAP pipeline",
)
@click.option(
    "--data",
    type=click.Path(path_type=Path),
    default=Path("data/mhc_ab.npz"),
    show_default=True,
    help="Path to training NPZ",
)
@click.option(
    "--netmhcpan",
    type=click.Path(path_type=Path),
    default=None,
    help="NetMHCpan install path",
)
@click.option(
    "--n_background",
    type=int,
    default=DEFAULT_BACKGROUND_SIZE,
    show_default=True,
    help="Background set size per bucket",
)
@click.option(
    "--n_foreground",
    type=int,
    default=DEFAULT_FOREGROUND_SIZE,
    show_default=True,
    help="Foreground set size per bucket",
)
@click.option(
    "--nsamples",
    type=int,
    default=DEFAULT_NSAMPLES,
    show_default=True,
    help="KernelSHAP sample budget per peptide",
)
@click.option(
    "--netmhcpan_jobs",
    type=int,
    default=None,
    help=(
        "Parallel NetMHCpan subprocesses "
        "(default: uses $NETMHCPAN_JOBS or 1)"
    ),
)
@click.option(
    "--alleles",
    default=None,
    help="Comma-separated alleles (default: auto top-N)",
)
@click.option(
    "--max_alleles",
    type=int,
    default=10,
    show_default=True,
    help="Max alleles to process when auto-selecting",
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path),
    default=Path("explainer"),
    show_default=True,
    help="Output directory",
)
def cli(
    data: Path,
    netmhcpan: Path | None,
    n_background: int,
    n_foreground: int,
    nsamples: int,
    netmhcpan_jobs: int | None,
    alleles: str | None,
    max_alleles: int,
    output_dir: Path,
) -> None:
    """Click CLI wrapper for the SHAP pipeline."""
    main(
        data=data,
        netmhcpan=netmhcpan,
        n_background=n_background,
        n_foreground=n_foreground,
        nsamples=nsamples,
        netmhcpan_jobs=netmhcpan_jobs,
        alleles=alleles,
        max_alleles=max_alleles,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter,missing-kwoa
