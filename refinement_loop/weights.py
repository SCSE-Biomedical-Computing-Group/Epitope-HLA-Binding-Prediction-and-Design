"""SHAP weight loading utilities for refinement."""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from generator.logging_utils import get_run_logger


LOGGER = get_run_logger(__name__)


def load_shap_weights(
    shap_json_path: Optional[Path] = None,
    ablation_npz_path: Optional[Path] = None,
    allele: Optional[str] = None,
    max_positions: int = 11,
) -> np.ndarray:
    """Load per-position importance weights from SHAP artifacts."""
    _ = ablation_npz_path  # kept for API compatibility
    weights = np.ones(max_positions) / max_positions

    if shap_json_path and Path(shap_json_path).exists():
        try:
            with open(shap_json_path) as f:
                results = json.load(f)

            # New SHAP format keyed by allele
            if allele and allele in results:
                pw = results[allele].get("position_weights")
                if pw:
                    arr = np.array(pw[:max_positions], dtype=np.float64)
                    total = arr.sum()
                    if total > 0:
                        weights = arr / total
                    LOGGER.info(
                        "Loaded SHAP weights for %s from %s",
                        allele,
                        shap_json_path,
                    )
                    return weights

            # Fallback: average position weights across all alleles
            all_pw = []
            for _, val in results.items():
                if isinstance(val, dict) and "position_weights" in val:
                    all_pw.append(
                        np.array(val["position_weights"]
                                 [:max_positions], dtype=np.float64)
                    )
            if all_pw:
                avg = np.mean(all_pw, axis=0)
                total = avg.sum()
                if total > 0:
                    weights = avg / total
                LOGGER.info(
                    "Loaded average SHAP weights from %s",
                    shap_json_path,
                )
                return weights

        except Exception as exc:
            LOGGER.warning("Could not load SHAP JSON: %s", exc)

    LOGGER.info("Using uniform position weights")
    return weights


def load_xai_weights(
    xai_json_path: Optional[Path] = None,
    ablation_npz_path: Optional[Path] = None,
    allele: Optional[str] = None,
    max_positions: int = 11,
) -> np.ndarray:
    """Backward-compatible alias for ``load_shap_weights``."""
    return load_shap_weights(
        shap_json_path=xai_json_path,
        ablation_npz_path=ablation_npz_path,
        allele=allele,
        max_positions=max_positions,
    )
