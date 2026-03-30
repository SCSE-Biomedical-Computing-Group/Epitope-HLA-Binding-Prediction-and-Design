"""
End-to-end generation pipeline orchestration.
"""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

from generator.ar_model import create_ar_model
from generator.checkpoints import export_ar_preset, get_checkpoint_upgrade_fn
from generator.logging_utils import configure_run_logger, get_run_logger
from generator.refinement_loop.loop import batch_refinement
from generator.refinement_loop.weights import load_shap_weights
from generator.scoring.scoring_wrapper import PredictorWrapper
from generator.selection.diversity import (
    compute_diversity_stats,
    compute_novelty_stats,
    compute_uniqueness_stats,
    motif_analysis,
    progressive_diverse_selection,
    progressive_diverse_selection_with_quotas,
)
from generator.training import (
    ARGenerator,
    sample_diverse_candidates,
    stratified_train_val_split,
    train_ar_model,
)
from generator.utils import (
    get_unique_alleles,
    load_config,
    load_mhc_data,
    save_results,
    save_run_summary,
    set_seed,
)
from evaluation.metrics import evaluate_binding_pass_rates


LOGGER = get_run_logger(__name__)


def _uniform_position_weights(max_positions: int = 11) -> np.ndarray:
    """Return uniform position weights when refinement runs without SHAP."""
    return np.ones(max_positions, dtype=np.float64) / max_positions


def _length_proportions(length_weights: str | None) -> Dict[int, float]:
    """Return normalized per-length proportions for 8/9/10/11-mers."""
    if not length_weights:
        return {8: 0.10, 9: 0.40, 10: 0.30, 11: 0.20}

    parsed = {8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}
    for chunk in length_weights.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            length_text, weight_text = chunk.split(":", maxsplit=1)
            length = int(length_text.strip())
            weight = float(weight_text.strip())
        except ValueError as exc:
            raise ValueError(
                "length_weights must look like "
                '"8:1,9:6,10:2,11:1"'
            ) from exc
        if length not in parsed:
            raise ValueError(
                "length_weights only supports peptide lengths 8, 9, 10, 11"
            )
        if weight < 0:
            raise ValueError("length_weights values must be non-negative")
        parsed[length] = weight

    total = sum(parsed.values())
    if total <= 0:
        raise ValueError("length_weights must contain at least one positive value")

    return {length: weight / total for length, weight in parsed.items()}


def _length_distribution(
    length_weights: str | None,
    use_length_quotas: bool,
) -> Dict[int, float]:
    """Return the candidate-generation distribution for peptide lengths."""
    if not use_length_quotas:
        return {8: 0.25, 9: 0.25, 10: 0.25, 11: 0.25}
    return _length_proportions(length_weights)


def _quota_counts(num_final: int, length_weights: str | None) -> Dict[int, int]:
    """Convert configured proportions into exact integer final-count quotas."""
    proportions = _length_proportions(length_weights)
    raw_counts = {
        length: num_final * proportion
        for length, proportion in proportions.items()
    }
    counts = {
        length: int(np.floor(value))
        for length, value in raw_counts.items()
    }
    remainder = num_final - sum(counts.values())

    if remainder > 0:
        priority = sorted(
            raw_counts,
            key=lambda length: (raw_counts[length] - counts[length], counts[length]),
            reverse=True,
        )
        for idx in range(remainder):
            counts[priority[idx % len(priority)]] += 1

    return counts


def _uses_shap_guidance(refine_mode: str) -> bool:
    """Return whether the selected refinement mode uses SHAP guidance."""
    return refine_mode in {"shap", "xai"}


def _get_shap_json_path(args: SimpleNamespace) -> Path | None:
    """Return the configured SHAP JSON path, honoring legacy arg names."""
    if hasattr(args, "shap_json"):
        return args.shap_json
    return getattr(args, "xai_json", None)


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "mps":
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    if device_str == "auto":
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(device_str)


def generate_for_allele(
    allele: str,
    ar_model: ARGenerator,
    predictor: PredictorWrapper,
    shap_weights: np.ndarray,
    train_peptides: List[str],
    args: SimpleNamespace,
    device: torch.device,
) -> Tuple[List[Dict], List[Dict], Dict, Dict]:
    """
    Generate peptides for a single allele.
    """
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Generating for %s", allele)
    LOGGER.info("%s", "=" * 60)

    length_dist = _length_distribution(
        args.length_weights,
        args.length_quotas,
    )
    LOGGER.info(
        "Length weights: %s",
        args.length_weights if args.length_weights else "default",
    )
    LOGGER.info("Candidate length mix: %s", length_dist)

    LOGGER.info("[1/4] Generating candidate peptides...")
    target_candidates = args.candidate_pool_size

    candidates = sample_diverse_candidates(
        ar_model,
        allele,
        target_candidates,
        length_dist,
        device,
        temperature_schedule=[0.7, 0.9, 1.1, 1.3],
        top_p_schedule=[0.85, 0.9, 0.95],
        batch_size=64,
        verbose=args.verbose,
    )

    LOGGER.info("Generated %s unique candidates", len(candidates))

    LOGGER.info("[2/4] Scoring candidates with predictor...")
    alleles_list = [candidate["allele"] for candidate in candidates]
    peptides_list = [candidate["peptide"] for candidate in candidates]

    scores = predictor.score(alleles_list, peptides_list)
    for index, candidate in enumerate(candidates):
        candidate["score"] = float(scores[index])

    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True,
    )

    top_scores = [candidate["score"] for candidate in candidates[:10]]
    LOGGER.info(
        "Top-10 scores: %.4f +/- %.4f",
        np.mean(top_scores),
        np.std(top_scores),
    )

    peptides_before_refinement = [
        candidate["peptide"] for candidate in candidates
    ]

    if args.enable_refinement:
        if _uses_shap_guidance(args.refine_mode):
            LOGGER.info("[3/4] Refining top candidates with SHAP guidance...")
        else:
            LOGGER.info(
                "[3/4] Refining top candidates without SHAP guidance..."
            )

        top_candidates = candidates[: min(200, len(candidates))]

        def batch_score_fn(
            alleles: List[str],
            peptides: List[str],
        ) -> np.ndarray:
            return predictor.score(alleles, peptides)

        refined = batch_refinement(
            top_candidates,
            batch_score_fn,
            shap_weights,
            n_steps=args.refine_steps,
            mutation_mode="controlled",
            batch_size=32,
            verbose=args.verbose,
        )

        candidates = refined + candidates[200:]
        candidates = sorted(
            candidates,
            key=lambda item: item["score"],
            reverse=True,
        )

        improved = sum(
            1
            for candidate in refined
            if candidate.get("score_improvement", 0) > 0
        )
        LOGGER.info(
            "Improved %s/%s candidates through refinement",
            improved,
            len(refined),
        )

        peptides_after_refinement = [
            candidate["peptide"] for candidate in candidates
        ]
    else:
        LOGGER.info("[3/4] Skipping refinement (disabled)")
        peptides_after_refinement = peptides_before_refinement

    uniqueness_stats = compute_uniqueness_stats(
        peptides_before_refinement,
        peptides_after_refinement,
    )
    LOGGER.info(
        "Uniqueness: %s/%s (%.1f%%)",
        uniqueness_stats["unique_after"],
        uniqueness_stats["total_after"],
        uniqueness_stats["uniqueness_rate_after"] * 100,
    )
    if args.enable_refinement:
        LOGGER.info(
            "Duplicates: %s -> %s",
            uniqueness_stats["duplicates_before"],
            uniqueness_stats["duplicates_after"],
        )

    LOGGER.info("[4/4] Selecting final diverse set...")
    top_nodiv = candidates[: min(args.num_final, len(candidates))]

    if args.length_quotas:
        quota_counts = _quota_counts(args.num_final, args.length_weights)
        LOGGER.info("Final length quotas: %s", quota_counts)
        final = progressive_diverse_selection_with_quotas(
            candidates,
            args.num_final,
            quota_counts,
            min_distance_schedule=[6, 5, 4, 3],
            max_kmer_jaccard=args.max_jaccard,
            verbose=args.verbose,
        )
    else:
        final = progressive_diverse_selection(
            candidates,
            args.num_final,
            min_distance_schedule=[6, 5, 4, 3],
            max_kmer_jaccard=args.max_jaccard,
            verbose=args.verbose,
        )

    final_peptides = [candidate["peptide"] for candidate in final]
    final_scores = [candidate["score"] for candidate in final]
    diversity_stats = compute_diversity_stats(final_peptides)
    novelty_stats = compute_novelty_stats(final_peptides, train_peptides)

    LOGGER.info("Final selection: %s peptides", len(final))
    LOGGER.info(
        "Score: %.4f +/- %.4f",
        np.mean(final_scores),
        np.std(final_scores),
    )
    LOGGER.info(
        "Avg pairwise edit distance: %.2f",
        diversity_stats["avg_pairwise_edit_distance"],
    )
    LOGGER.info(
        "Length distribution: %s",
        diversity_stats["length_distribution"],
    )
    LOGGER.info(
        "Novelty: mean NN dist=%.2f, median=%.1f, min=%s, exact_matches=%s",
        novelty_stats["mean_nn_distance"],
        novelty_stats["median_nn_distance"],
        novelty_stats["min_nn_distance"],
        novelty_stats["num_exact_matches"],
    )

    return final, top_nodiv, uniqueness_stats, novelty_stats


def run_pipeline(args: SimpleNamespace) -> None:
    """Run the generator pipeline with file-backed logging."""
    if args.config:
        config = load_config(args.config)
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    else:
        config = vars(args).copy()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    config["log_file"] = str(log_path)
    configure_run_logger(log_path)

    with open(log_path, "a", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            try:
                _run_pipeline_core(args, config, run_dir, run_id)
            except Exception:
                LOGGER.exception("Pipeline failed.")
                raise SystemExit(1) from None


def _run_pipeline_core(
    args: SimpleNamespace,
    config: Dict,
    run_dir: Path,
    run_id: str,
) -> None:
    """Run the generator pipeline after log redirection is configured."""
    if args.seed is not None:
        set_seed(args.seed)
        LOGGER.info("Random seed: %s", args.seed)

    device = get_device(args.device)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Run ID: %s", run_id)

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {key: str(value) if isinstance(value, Path)
             else value for key, value in config.items()},
            f,
            indent=2,
        )

    LOGGER.info("Loading data from %s...", args.data)
    alleles, peptides, _ = load_mhc_data(
        args.data, filter_lengths=[8, 9, 10, 11]
    )
    LOGGER.info("Loaded %s peptides", len(peptides))

    all_unique_alleles = get_unique_alleles(alleles)
    LOGGER.info("Unique alleles: %s", len(all_unique_alleles))

    if args.alleles.lower() == "all":
        counts = Counter(alleles)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        target_alleles = [allele for allele,
                          _count in ranked[: args.max_alleles]]
        LOGGER.info(
            "Using top %s alleles by frequency: %s",
            len(target_alleles),
            target_alleles,
        )
    else:
        target_alleles = [allele.strip() for allele in args.alleles.split(",")]
        for allele in target_alleles:
            if allele not in all_unique_alleles:
                LOGGER.warning("Allele %s not found in training data", allele)

    ar_model = create_ar_model(embed_dim=args.ar_embed_dim)
    upgrade_legacy_ar_checkpoint = get_checkpoint_upgrade_fn()

    if args.ar_ckpt and args.ar_ckpt.exists():
        LOGGER.info("Loading pre-trained AR model from %s...", args.ar_ckpt)
        try:
            state_dict = torch.load(args.ar_ckpt, map_location=device)
            ar_model.load_state_dict(upgrade_legacy_ar_checkpoint(state_dict))
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load the checkpoint into the class-I AR "
                "generator. Use a compatible Transformer checkpoint "
                "or retrain without --ar_ckpt."
            ) from exc
    else:
        LOGGER.info("Training AR model (transformer)...")

        split_seed = args.seed if args.seed is not None else 0
        train_data, val_data = stratified_train_val_split(
            alleles,
            peptides,
            val_fraction=args.val_fraction,
            seed=split_seed,
        )
        LOGGER.info(
            "Stratified split: train=%s, val=%s",
            len(train_data[1]),
            len(val_data[1]),
        )

        ar_save_path = run_dir / "ar_model.pt"
        train_ar_model(
            ar_model,
            train_data,
            val_data,
            epochs=args.ar_epochs,
            batch_size=args.ar_batch_size,
            lr=args.ar_lr,
            device=device,
            save_path=ar_save_path,
            verbose=args.verbose,
            early_stopping_patience=(
                args.early_stop_patience
                if args.early_stop_patience > 0
                else None
            ),
            early_stopping_min_delta=args.early_stop_min_delta,
            min_epochs_before_stop=args.early_stop_min_epochs,
        )

        LOGGER.info("AR model saved to %s", ar_save_path)
        export_ar_preset(
            ar_save_path,
            args.save_ar_preset,
            run_id,
            args,
            train_size=len(train_data[1]),
            val_size=len(val_data[1]),
        )

    ar_model = ar_model.to(device)
    ar_model.eval()

    LOGGER.info("Loading predictor (NetMHCpan 4.2)...")
    predictor = PredictorWrapper(netmhcpan_path=args.netmhcpan)

    shap_json_path = _get_shap_json_path(args)
    if _uses_shap_guidance(args.refine_mode):
        LOGGER.info("Loading SHAP importance weights...")
        default_position_weights = load_shap_weights(
            shap_json_path=shap_json_path
        )
    else:
        LOGGER.info("Using uniform position weights (without SHAP)...")
        default_position_weights = _uniform_position_weights()
    LOGGER.info("Position weights: %s", default_position_weights[:11].round(3))

    all_summaries = []

    for allele in target_alleles:
        try:
            allele_train_peptides = [
                peptide
                for allele_name, peptide in zip(alleles, peptides)
                if allele_name == allele
            ]
            if not allele_train_peptides:
                allele_train_peptides = peptides

            if _uses_shap_guidance(args.refine_mode):
                allele_shap_weights = load_shap_weights(
                    shap_json_path=shap_json_path,
                    allele=allele,
                )
            else:
                allele_shap_weights = default_position_weights.copy()

            (
                final_peptides,
                top_nodiv,
                uniqueness_stats,
                novelty_stats,
            ) = generate_for_allele(
                allele,
                ar_model,
                predictor,
                allele_shap_weights,
                allele_train_peptides,
                args,
                device,
            )

            save_results(
                final_peptides,
                args.out_dir,
                allele,
                run_id,
                suffix="div",
            )
            save_results(
                top_nodiv,
                args.out_dir,
                allele,
                run_id,
                suffix="nodiv",
            )

            peptides_list = [item["peptide"] for item in final_peptides]
            scores_list = [item["score"] for item in final_peptides]

            binding_rates = evaluate_binding_pass_rates(
                peptides_list,
                scores_list,
                allele,
            )

            summary = {
                "allele": allele,
                "total_peptides": len(final_peptides),
                "score_stats": {
                    "mean": float(np.mean(scores_list)),
                    "max": float(np.max(scores_list)),
                    "min": float(np.min(scores_list)),
                },
                "binding_pass_rates": binding_rates,
                "diversity_stats": compute_diversity_stats(peptides_list),
                "novelty_stats": novelty_stats,
                "uniqueness_stats": uniqueness_stats,
                "motifs": motif_analysis(peptides_list, positions=[2, -1]),
            }
            all_summaries.append(summary)

        except Exception:
            LOGGER.exception("Error generating for %s", allele)

    save_run_summary(all_summaries, args.out_dir, run_id, config)

    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Generation complete!")
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Run ID: %s", run_id)
    LOGGER.info("Output directory: %s", run_dir)
    LOGGER.info("Alleles processed: %s", len(all_summaries))
    LOGGER.info(
        "Total peptides: %s",
        sum(summary["total_peptides"] for summary in all_summaries),
    )

    binding_rate_summaries = [
        summary.get("binding_pass_rates", {})
        for summary in all_summaries
    ]
    binding_rate_summaries = [
        binding_rates
        for binding_rates in binding_rate_summaries
        if "pct_SB" in binding_rates
    ]
    if binding_rate_summaries:
        avg_sb = np.mean([binding_rates["pct_SB"]
                         for binding_rates in binding_rate_summaries])
        avg_wb = np.mean([binding_rates["pct_WB"]
                         for binding_rates in binding_rate_summaries])
        LOGGER.info(
            "Aggregate binding pass-rates (macro-avg over %s alleles):",
            len(binding_rate_summaries),
        )
        LOGGER.info("%%SB  (IC50 <= 50 nM): %.1f%%", avg_sb)
        LOGGER.info("%%WB  (IC50 <= 500 nM): %.1f%%", avg_wb)
