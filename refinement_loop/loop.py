"""Refinement loop for SHAP-guided peptide optimization."""

import random
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from generator.logging_utils import get_run_logger

# Standard amino acids (excluding special tokens)
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
LOGGER = get_run_logger(__name__)


def get_position_weights(
    peptide_length: int,
    base_weights: np.ndarray,
    anchor_boost: float = 1.5,
) -> np.ndarray:
    """
    Get position weights normalized for a specific peptide length.
    """
    weights = base_weights[:peptide_length].copy()

    # Boost anchor positions
    if peptide_length >= 2:
        weights[1] *= anchor_boost  # P2
    weights[-1] *= anchor_boost  # C-term

    # Normalize
    weights = weights / weights.sum()
    return weights


def get_similar_aas(aa: str) -> List[str]:
    """Get amino acids with similar properties."""
    hydrophobic = set("AILMFVPWG")
    polar = set("STYCNQ")
    charged_pos = set("KRH")
    charged_neg = set("DE")
    aromatic = set("FYW")

    similar = []
    if aa in hydrophobic:
        similar.extend(hydrophobic)
    if aa in polar:
        similar.extend(polar)
    if aa in charged_pos:
        similar.extend(charged_pos)
    if aa in charged_neg:
        similar.extend(charged_neg)
    if aa in aromatic:
        similar.extend(aromatic)

    return list(set(similar)) if similar else STANDARD_AAS


def propose_mutation(
    peptide: str,
    position_weights: np.ndarray,
    mutation_mode: str = "controlled",
) -> Tuple[str, int, str, str]:
    """
    Propose a single amino acid mutation guided by position weights.
    """
    length = len(peptide)
    weights = position_weights[:length]
    weights = weights / weights.sum()

    # Sample position proportional to importance
    position = np.random.choice(length, p=weights)
    original_aa = peptide[position]

    if mutation_mode == "alanine":
        new_aa = "A" if original_aa != "A" else "G"
    elif mutation_mode == "conservative":
        similar_aas = get_similar_aas(original_aa)
        candidates = [aa for aa in similar_aas if aa != original_aa]
        new_aa = random.choice(
            candidates) if candidates else random.choice(STANDARD_AAS)
    else:  # controlled/random
        candidates = [aa for aa in STANDARD_AAS if aa != original_aa]
        new_aa = random.choice(candidates)

    mutated = peptide[:position] + new_aa + peptide[position + 1:]
    return mutated, position, original_aa, new_aa


def hill_climb_refinement(
    candidate: Dict,
    score_fn: Callable[[str, str], float],
    position_weights: np.ndarray,
    n_steps: int = 15,
    mutation_mode: str = "controlled",
    allow_downhill: float = 0.0,
) -> Dict:
    """Refine a single peptide using hill-climbing with SHAP-guided mutations."""
    allele = candidate["allele"]
    peptide = candidate["peptide"]
    score = candidate.get("score", score_fn(allele, peptide))

    best_peptide = peptide
    best_score = score

    # Get position weights for this peptide length
    pep_len = len(peptide)
    weights = get_position_weights(pep_len, position_weights)

    mutation_history = []

    for step in range(n_steps):
        mutated, pos, old_aa, new_aa = propose_mutation(
            peptide, weights, mutation_mode)
        new_score = score_fn(allele, mutated)

        accept = False
        if new_score > score:
            accept = True
        elif allow_downhill > 0 and random.random() < allow_downhill:
            accept = True

        if accept:
            peptide = mutated
            score = new_score
            mutation_history.append(
                {
                    "step": step,
                    "position": pos,
                    "from": old_aa,
                    "to": new_aa,
                    "score_delta": new_score - score,
                }
            )

            if score > best_score:
                best_peptide = peptide
                best_score = score

    refined = candidate.copy()
    refined["peptide"] = best_peptide
    refined["score"] = best_score
    refined["source"] = "AR+SHAP"
    refined["seed_peptide"] = candidate["peptide"]
    refined["n_mutations"] = len(mutation_history)
    refined["score_improvement"] = best_score - candidate.get("score", 0)

    return refined


def batch_refinement(
    candidates: List[Dict],
    score_fn: Callable[[List[str], List[str]], np.ndarray],
    position_weights: np.ndarray,
    n_steps: int = 15,
    mutation_mode: str = "controlled",
    batch_size: int = 32,
    verbose: bool = True,
) -> List[Dict]:
    """
    Refine multiple peptides in batches.
    """
    _ = batch_size  # kept for API compatibility

    alleles = [c["allele"] for c in candidates]
    peptides = [c["peptide"] for c in candidates]

    if verbose:
        LOGGER.info("Scoring %s candidates...", len(candidates))

    scores = score_fn(alleles, peptides)
    for i, candidate in enumerate(candidates):
        candidate["score"] = scores[i]

    refined = []
    pbar = tqdm(
        candidates,
        desc="  Refining",
        disable=not verbose or not sys.stderr.isatty(),
    )

    for candidate in pbar:
        def single_score_fn(allele: str, peptide: str) -> float:
            return float(score_fn([allele], [peptide])[0])

        refined_candidate = hill_climb_refinement(
            candidate,
            single_score_fn,
            position_weights,
            n_steps=n_steps,
            mutation_mode=mutation_mode,
        )
        refined.append(refined_candidate)

    return refined
