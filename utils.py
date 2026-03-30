"""
I/O utilities for the generator pipeline.
"""

from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from generator.logging_utils import get_run_logger
from .selection.diversity import compute_diversity_stats, motif_analysis

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
LOGGER = get_run_logger(__name__)


def load_mhc_data(
    data_path: Path,
    filter_lengths: Optional[List[int]] = None
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Load HLA-peptide binding data from npz file.

    Args:
        data_path: path to .npz file
        filter_lengths: optional list of peptide lengths to keep

    Returns:
        (alleles, peptides, targets) tuple
    """
    data = np.load(data_path, allow_pickle=True)

    required_keys = ("allele", "peptide", "measurement_value")
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        missing = ", ".join(repr(key) for key in missing_keys)
        raise KeyError(
            f"Missing required key(s) in {data_path}: {missing}. "
            "Expected canonical NPZ schema: "
            "'allele', 'peptide', 'measurement_value'."
        )

    alleles = data["allele"]
    peptides = data["peptide"]
    targets = data["measurement_value"]

    # Convert to lists if needed
    if isinstance(alleles, np.ndarray):
        alleles = [str(a) for a in alleles]
    if isinstance(peptides, np.ndarray):
        peptides = [str(p) for p in peptides]

    # Filter by length if specified
    if filter_lengths:
        mask = [len(p) in filter_lengths for p in peptides]
        alleles = [a for a, m in zip(alleles, mask) if m]
        peptides = [p for p, m in zip(peptides, mask) if m]
        targets = (
            targets[mask]
            if hasattr(targets, '__getitem__')
            else np.array([t for t, m in zip(targets, mask) if m])
        )

    return alleles, peptides, targets


def get_unique_alleles(alleles: List[str]) -> List[str]:
    """Get unique alleles sorted alphabetically."""
    return sorted(set(alleles))


def save_results(
    results: List[Dict],
    output_dir: Path,
    allele: str,
    run_id: str,
    suffix: str = 'final'
) -> None:
    """
    Save generation results for a single allele.

    Args:
        results: list of result dicts
        output_dir: base output directory
        allele: HLA allele name
        run_id: run identifier
        suffix: file suffix ('div', 'nodiv', 'final')
    """
    # Clean allele name for directory
    allele_clean = allele.replace('*', '_').replace(':', '_')
    allele_dir = output_dir / run_id / allele_clean
    allele_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = allele_dir / f'final_peptides_{suffix}.csv'
    df.to_csv(csv_path, index=False)

    # Compute summary stats
    peptides = [r['peptide'] for r in results]
    scores = [r.get('score', 0) for r in results]

    diversity_stats = compute_diversity_stats(peptides)
    motifs = motif_analysis(peptides, positions=[2, -1])  # P2 and C-term

    summary = {
        'allele': allele,
        'suffix': suffix,
        'total_peptides': int(len(results)),
        'score_stats': {
            'mean': float(np.mean(scores)) if scores else 0,
            'median': float(np.median(scores)) if scores else 0,
            'max': float(np.max(scores)) if scores else 0,
            'min': float(np.min(scores)) if scores else 0,
            'std': float(np.std(scores)) if scores else 0
        },
        'diversity_stats': diversity_stats,
        'motifs': motifs,
        'source_distribution': {
            str(key): int(value)
            for key, value in pd.Series(
                [r.get('source', 'AR') for r in results]
            ).value_counts().items()
        },
    }

    json_path = allele_dir / f'summary_{suffix}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    LOGGER.info("Saved %s peptides to %s", len(results), csv_path)


def save_run_summary(
    all_summaries: List[Dict],
    output_dir: Path,
    run_id: str,
    config: Dict
) -> None:
    """
    Save overall run summary.

    Args:
        all_summaries: list of per-allele summaries
        output_dir: base output directory
        run_id: run identifier
        config: run configuration
    """
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Convert config values to JSON-serializable types
    json_config = {}
    for k, v in config.items():
        if isinstance(v, Path):
            json_config[k] = str(v)
        elif hasattr(v, 'item'):  # numpy scalar
            json_config[k] = v.item()
        else:
            json_config[k] = v

    run_summary = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'config': json_config,
        'num_alleles': len(all_summaries),
        'total_peptides': sum(
            summary.get('total_peptides', 0)
            for summary in all_summaries
        ),
        'allele_summaries': all_summaries
    }

    summary_path = run_dir / 'run_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2, default=str)

    LOGGER.info("Run summary saved to %s", summary_path)


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from JSON file or return defaults."""
    defaults = {
        'ar_embed_dim': 128,
        'ar_num_layers': 3,
        'ar_epochs': 50,
        'ar_batch_size': 64,
        'ar_lr': 1e-3,
        'gen_temperature': 1.0,
        'gen_top_p': 0.9,
        'gen_samples_per_length': 100,
        'refine_enabled': True,
        'refine_n_steps': 15,
        'refine_mutation_mode': 'controlled',
        'num_final_per_allele': 50,
        'diversity_min_edit_dist': 2,
        'diversity_max_jaccard': 0.7,
        'length_quotas': {8: 0.10, 9: 0.40, 10: 0.30, 11: 0.20},
        'use_length_quotas': True
    }

    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = json.load(f)
        defaults.update(user_config)

    return defaults


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have separate seed, but set deterministic mode
        pass
