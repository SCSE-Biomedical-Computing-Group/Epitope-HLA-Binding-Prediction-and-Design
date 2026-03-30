"""
Diversity filtering for peptide selection.

Provides edit distance and k-mer based diversity metrics,
plus greedy selection algorithms to ensure diverse final sets.
"""

from collections import Counter
from typing import Dict, List, Optional, Set

import numpy as np

from generator.logging_utils import get_run_logger


LOGGER = get_run_logger(__name__)


def edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def hamming_distance(s1: str, s2: str) -> int:
    """
    Compute Hamming distance between two strings of equal length.
    Returns max possible distance if lengths differ.
    """
    if len(s1) != len(s2):
        return max(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def get_kmers(sequence: str, k: int = 3) -> Set[str]:
    """
    Extract all k-mers from a sequence.

    Args:
        sequence: input string
        k: k-mer length

    Returns:
        set of k-mer strings
    """
    if len(sequence) < k:
        return {sequence}
    return {sequence[i:i+k] for i in range(len(sequence) - k + 1)}


def kmer_jaccard(s1: str, s2: str, k: int = 3) -> float:
    """
    Compute k-mer Jaccard similarity between two sequences.
    """
    kmers1 = get_kmers(s1, k)
    kmers2 = get_kmers(s2, k)

    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)

    if union == 0:
        return 1.0
    return intersection / union


def is_sufficiently_diverse(
    peptide: str,
    selected: List[str],
    min_edit_distance: int = 2,
    max_kmer_jaccard: float = 0.7,
    k: int = 3
) -> bool:
    """
    Check if a peptide is sufficiently diverse from already selected ones.
    """
    if not selected:
        return True

    for existing in selected:
        # Check edit distance
        if edit_distance(peptide, existing) < min_edit_distance:
            return False

        # Check k-mer similarity (only for same-length peptides)
        if len(peptide) == len(existing):
            if kmer_jaccard(peptide, existing, k) > max_kmer_jaccard:
                return False

    return True


def greedy_diverse_selection(
    candidates: List[Dict],
    n_select: int,
    min_edit_distance: int = 2,
    max_kmer_jaccard: float = 0.7,
    k: int = 3,
    length_quotas: Optional[Dict[int, int]] = None,
    score_key: str = 'score'
) -> List[Dict]:
    """
    Greedily select diverse peptides from candidates.
    """
    # Sort by score descending
    sorted_candidates = sorted(
        candidates, key=lambda x: x.get(score_key, 0), reverse=True)

    selected = []
    selected_peptides = []
    length_counts = {8: 0, 9: 0, 10: 0, 11: 0}

    for candidate in sorted_candidates:
        if len(selected) >= n_select:
            break

        peptide = candidate['peptide']
        length = len(peptide)

        # Check length quota
        if length_quotas and length in length_quotas:
            if length_counts.get(length, 0) >= length_quotas[length]:
                continue

        # Check diversity
        if is_sufficiently_diverse(
            peptide, selected_peptides,
            min_edit_distance, max_kmer_jaccard, k
        ):
            selected.append(candidate)
            selected_peptides.append(peptide)
            length_counts[length] = length_counts.get(length, 0) + 1

    return selected


def progressive_diverse_selection(
    candidates: List[Dict],
    n_select: int,
    min_distance_schedule: List[int] = [6, 5, 4, 3],
    max_kmer_jaccard: float = 0.7,
    k: int = 3,
    score_key: str = 'score',
    verbose: bool = False
) -> List[Dict]:
    """
    Progressively relax diversity constraints until target count is reached.
    """
    for min_dist in min_distance_schedule:
        selected = greedy_diverse_selection(
            candidates, n_select,
            min_edit_distance=min_dist,
            max_kmer_jaccard=max_kmer_jaccard,
            k=k,
            score_key=score_key
        )

        if verbose:
            LOGGER.info(
                "min_dist=%s: selected %s peptides",
                min_dist,
                len(selected),
            )

        if len(selected) >= n_select:
            return selected[:n_select]

    # Fallback: take top n without any diversity filter
    if verbose:
        LOGGER.info(
            "Fallback: no diversity filter, taking top %s",
            n_select,
        )

    sorted_candidates = sorted(
        candidates, key=lambda x: x.get(score_key, 0), reverse=True)
    return sorted_candidates[:n_select]


def progressive_diverse_selection_with_quotas(
    candidates: List[Dict],
    n_select: int,
    quota_counts: Dict[int, int],
    min_distance_schedule: List[int] = [6, 5, 4, 3],
    max_kmer_jaccard: float = 0.7,
    k: int = 3,
    score_key: str = 'score',
    verbose: bool = False
) -> List[Dict]:
    """
    Progressively relax diversity constraints with soft quotas and backfill.
    """
    # Group candidates by length
    by_length = {8: [], 9: [], 10: [], 11: []}
    for c in candidates:
        length = len(c['peptide'])
        if length in by_length:
            by_length[length].append(c)

    # Sort each group by score
    for length in by_length:
        by_length[length] = sorted(
            by_length[length], key=lambda x: x.get(score_key, 0), reverse=True)

    selected = []
    selected_peptides = []
    length_counts = {8: 0, 9: 0, 10: 0, 11: 0}

    # Try each min_distance threshold
    for min_dist in min_distance_schedule:
        # Try to fill quotas for each length
        for length in sorted(
            quota_counts.keys(),
            key=lambda value: quota_counts[value],
            reverse=True,
        ):
            target = quota_counts[length]
            current = length_counts[length]

            if current >= target:
                continue

            # Try to add more of this length
            for candidate in by_length[length]:
                if candidate in selected:
                    continue

                peptide = candidate['peptide']

                # Check diversity
                if is_sufficiently_diverse(
                    peptide, selected_peptides,
                    min_dist, max_kmer_jaccard, k
                ):
                    selected.append(candidate)
                    selected_peptides.append(peptide)
                    length_counts[length] += 1

                    if length_counts[length] >= target:
                        break

        if verbose:
            LOGGER.info(
                "min_dist=%s: %s (target: %s)",
                min_dist,
                length_counts,
                quota_counts,
            )

        if len(selected) >= n_select:
            return selected[:n_select]

    # Backfill: if quotas not met, take next best overall regardless of length
    if len(selected) < n_select:
        if verbose:
            LOGGER.info(
                "Backfilling: %s/%s selected",
                len(selected),
                n_select,
            )

        # Get all unselected candidates sorted by score
        unselected = [c for c in candidates if c not in selected]
        unselected = sorted(unselected, key=lambda x: x.get(
            score_key, 0), reverse=True)

        for candidate in unselected:
            if len(selected) >= n_select:
                break

            peptide = candidate['peptide']

            # Use most relaxed threshold for backfill
            min_dist_backfill = min(min_distance_schedule)
            if is_sufficiently_diverse(
                peptide, selected_peptides,
                min_dist_backfill, max_kmer_jaccard, k
            ):
                selected.append(candidate)
                selected_peptides.append(peptide)
                length_counts[len(peptide)] += 1

        if verbose:
            LOGGER.info("After backfill: %s", length_counts)

    return selected


def compute_diversity_stats(peptides: List[str]) -> Dict:
    """
    Compute diversity statistics for a set of peptides.
    """
    if len(peptides) < 2:
        return {
            'count': len(peptides),
            'avg_pairwise_edit_distance': 0.0,
            'min_pairwise_edit_distance': 0,
            'avg_pairwise_hamming': 0.0,
            'unique_kmers': 0,
            'length_distribution': {}
        }

    # Pairwise edit distances
    edit_distances = []
    hamming_distances = []

    for index, peptide in enumerate(peptides):
        for other_peptide in peptides[index + 1:]:
            edit_distances.append(edit_distance(peptide, other_peptide))
            if len(peptide) == len(other_peptide):
                hamming_distances.append(
                    hamming_distance(peptide, other_peptide)
                )

    # Unique k-mers
    all_kmers = set()
    for pep in peptides:
        all_kmers.update(get_kmers(pep, k=3))

    # Length distribution
    length_dist = Counter(len(p) for p in peptides)

    return {
        'count': int(len(peptides)),
        'avg_pairwise_edit_distance': (
            float(np.mean(edit_distances)) if edit_distances else 0.0
        ),
        'min_pairwise_edit_distance': (
            int(min(edit_distances)) if edit_distances else 0
        ),
        'max_pairwise_edit_distance': (
            int(max(edit_distances)) if edit_distances else 0
        ),
        'avg_pairwise_hamming': (
            float(np.mean(hamming_distances)) if hamming_distances else 0.0
        ),
        'unique_3mers': int(len(all_kmers)),
        'length_distribution': {int(k): int(v) for k, v in length_dist.items()}
    }


def compute_novelty_stats(
    generated_peptides: List[str],
    training_peptides: List[str]
) -> Dict:
    """
    Compute novelty metrics comparing generated peptides to training set.
    """
    if not generated_peptides or not training_peptides:
        return {
            'mean_nn_distance': 0.0,
            'median_nn_distance': 0.0,
            'min_nn_distance': 0,
            'max_nn_distance': 0,
            'num_exact_matches': 0
        }

    nn_distances = []
    exact_matches = 0

    for gen_pep in generated_peptides:
        # Find nearest neighbor in training set
        min_dist = float('inf')
        for train_pep in training_peptides:
            dist = edit_distance(gen_pep, train_pep)
            if dist < min_dist:
                min_dist = dist
            if dist == 0:
                exact_matches += 1
                break  # Exact match found

        nn_distances.append(min_dist)

    return {
        'mean_nn_distance': float(np.mean(nn_distances)),
        'median_nn_distance': float(np.median(nn_distances)),
        'min_nn_distance': int(min(nn_distances)),
        'max_nn_distance': int(max(nn_distances)),
        'num_exact_matches': int(exact_matches),
        'novelty_rate': (
            float(1.0 - exact_matches / len(generated_peptides))
            if generated_peptides
            else 0.0
        ),
    }


def compute_uniqueness_stats(
    peptides_before: List[str],
    peptides_after: List[str]
) -> Dict:
    """
    Compute uniqueness/collapse statistics before and after refinement.
    """
    unique_before = len(set(peptides_before))
    unique_after = len(set(peptides_after))

    duplicates_before = len(peptides_before) - unique_before
    duplicates_after = len(peptides_after) - unique_after

    return {
        'unique_before': int(unique_before),
        'unique_after': int(unique_after),
        'total_before': int(len(peptides_before)),
        'total_after': int(len(peptides_after)),
        'duplicates_before': int(duplicates_before),
        'duplicates_after': int(duplicates_after),
        'uniqueness_rate_before': (
            float(unique_before / len(peptides_before))
            if peptides_before
            else 0.0
        ),
        'uniqueness_rate_after': (
            float(unique_after / len(peptides_after))
            if peptides_after
            else 0.0
        ),
        'collapse_rate': (
            float((unique_before - unique_after) / unique_before)
            if unique_before > 0
            else 0.0
        ),
    }


def motif_analysis(
    peptides: List[str],
    positions: List[int] = [1, -1]
) -> Dict:
    """
    Analyze amino acid frequencies at specified positions.
    """
    motifs = {}

    for pos in positions:
        aa_counts = Counter()
        valid_count = 0

        for pep in peptides:
            try:
                if pos > 0:
                    idx = pos - 1  # Convert to 0-indexed
                else:
                    idx = pos  # Negative indexing works as-is

                if 0 <= (idx if idx >= 0 else len(pep) + idx) < len(pep):
                    aa = pep[idx]
                    aa_counts[aa] += 1
                    valid_count += 1
            except IndexError:
                continue

        if valid_count > 0:
            # Normalize to frequencies
            aa_freq = {aa: count / valid_count for aa,
                       count in aa_counts.items()}
            # Sort by frequency
            aa_freq = dict(sorted(aa_freq.items(), key=lambda x: -x[1]))
            motifs[f'P{pos}' if pos > 0 else f'P{pos}(C-term)'] = aa_freq
        else:
            motifs[f'P{pos}' if pos > 0 else f'P{pos}(C-term)'] = {}

    return motifs


def select_with_length_quotas(
    candidates: List[Dict],
    n_total: int,
    length_quotas: Dict[int, float],
    min_edit_distance: int = 2,
    score_key: str = 'score'
) -> List[Dict]:
    """
    Select peptides with proportional length quotas.
    """
    # Calculate absolute quotas
    quotas = {length: max(1, int(n_total * prop))
              for length, prop in length_quotas.items()}

    # Adjust to exactly hit n_total
    total_allocated = sum(quotas.values())
    if total_allocated < n_total:
        # Add to largest quota
        max_len = max(quotas.keys(), key=lambda k: quotas[k])
        quotas[max_len] += n_total - total_allocated
    elif total_allocated > n_total:
        # Remove from largest quota
        max_len = max(quotas.keys(), key=lambda k: quotas[k])
        quotas[max_len] -= total_allocated - n_total

    # Select using greedy diversity
    selected = greedy_diverse_selection(
        candidates, n_total,
        min_edit_distance=min_edit_distance,
        length_quotas=quotas,
        score_key=score_key
    )

    return selected
