#!/usr/bin/env python3
"""
Main file for the HLA class I peptide generator pipeline.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import click
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="HLA Class I Peptide Generator Pipeline",
    epilog=(
        "See generator/README.md for usage examples, features, and outputs."
    ),
)
@click.option(
    "--data",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to training data (.npz file)",
)
@click.option(
    "--out_dir",
    "out_dir",
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    show_default=True,
    help="Output directory",
)
@click.option(
    "--netmhcpan",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path to NetMHCpan 4.2 installation "
        "(auto-detected via $NETMHCPAN or sibling dir)"
    ),
)
@click.option(
    "--shap_json",
    "--xai_json",
    "shap_json",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to SHAP results JSON (explainer/shap_results.json)",
)
@click.option(
    "--ar_ckpt",
    "ar_ckpt",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to pre-trained AR model (skip training if provided)",
)
@click.option(
    "--save_ar_preset",
    "save_ar_preset",
    type=click.Path(path_type=Path),
    default=Path("generator/checkpoints/ar_transformer_latest.pt"),
    show_default=True,
    help="Stable path to copy the trained AR checkpoint for later reuse",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to JSON config file",
)
@click.option(
    "--alleles",
    default="all",
    show_default=True,
    help='Comma-separated alleles or "all"',
)
@click.option(
    "--max_alleles",
    "max_alleles",
    type=int,
    default=10,
    show_default=True,
    help="When --alleles=all, keep top-N most frequent alleles",
)
@click.option(
    "--ar_embed_dim",
    "ar_embed_dim",
    type=int,
    default=128,
    show_default=True,
    help="AR embedding dimension",
)
@click.option(
    "--ar_epochs",
    "ar_epochs",
    type=int,
    default=50,
    show_default=True,
    help="AR training epochs",
)
@click.option(
    "--ar_batch_size",
    "ar_batch_size",
    type=int,
    default=64,
    show_default=True,
    help="AR training batch size",
)
@click.option(
    "--ar_lr",
    "ar_lr",
    type=float,
    default=1e-3,
    show_default=True,
    help="AR learning rate",
)
@click.option(
    "--val_fraction",
    "val_fraction",
    type=float,
    default=0.1,
    show_default=True,
    help="Validation fraction for AR split",
)
@click.option(
    "--early_stop_patience",
    "early_stop_patience",
    type=int,
    default=8,
    show_default=True,
    help="Early stopping patience on val loss; <=0 disables",
)
@click.option(
    "--early_stop_min_delta",
    "early_stop_min_delta",
    type=float,
    default=1e-4,
    show_default=True,
    help="Minimum val-loss improvement to reset patience",
)
@click.option(
    "--early_stop_min_epochs",
    "early_stop_min_epochs",
    type=int,
    default=5,
    show_default=True,
    help="Minimum epochs before early stopping can trigger",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    show_default=True,
    help="Sampling temperature",
)
@click.option(
    "--top_p",
    "top_p",
    type=float,
    default=0.9,
    show_default=True,
    help="Nucleus sampling threshold",
)
@click.option(
    "--top_k",
    "top_k",
    type=int,
    default=0,
    show_default=True,
    help="Top-k sampling (0=disabled)",
)
@click.option(
    "--samples_per_length",
    "samples_per_length",
    type=int,
    default=3000,
    show_default=True,
    help="Samples per length per allele",
)
@click.option(
    "--candidate_pool_size",
    "candidate_pool_size",
    type=int,
    default=400,
    show_default=True,
    help=(
        "Candidate pool size before refinement "
        "(default target: 300-500 per length)"
    ),
)
@click.option(
    "--enable_refinement/--no_refinement",
    "enable_refinement",
    default=True,
    show_default=True,
    help="Enable SHAP-guided refinement",
)
@click.option(
    "--refine_steps",
    "refine_steps",
    type=int,
    default=15,
    show_default=True,
    help="Refinement steps per candidate",
)
@click.option(
    "--refine_mode",
    "refine_mode",
    type=click.Choice(
        ["shap", "without_shap", "xai", "without_xai"],
        case_sensitive=True,
    ),
    default="shap",
    show_default=True,
    help="Guidance mode for refinement",
)
@click.option(
    "--length_weights",
    "length_weights",
    default=None,
    help=(
        "Optional comma-separated weights for 8/9/10/11-mers, "
        'for example "8:1,9:6,10:2,11:1"'
    ),
)
@click.option(
    "--num_final",
    "num_final",
    type=int,
    default=50,
    show_default=True,
    help="Final peptides per allele",
)
@click.option(
    "--length_quotas/--no_length_quotas",
    "length_quotas",
    default=True,
    show_default=True,
    help="Use length quotas (8=10pct, 9=40pct, 10=30pct, 11=20pct)",
)
@click.option(
    "--min_edit_distance",
    "min_edit_distance",
    type=int,
    default=2,
    show_default=True,
    help="Minimum pairwise edit distance",
)
@click.option(
    "--max_jaccard",
    "max_jaccard",
    type=float,
    default=0.7,
    show_default=True,
    help="Maximum k-mer Jaccard similarity",
)
@click.option(
    "--device",
    default="mps",
    show_default=True,
    help="Device: cpu, mps, cuda, or auto",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--verbose/--quiet",
    "verbose",
    default=True,
    show_default=True,
    help="Enable or suppress verbose output",
)
def main(**kwargs):
    """Click entry point."""
    from generator.pipeline import run_pipeline

    run_pipeline(SimpleNamespace(**kwargs))


if __name__ == '__main__':
    main()
