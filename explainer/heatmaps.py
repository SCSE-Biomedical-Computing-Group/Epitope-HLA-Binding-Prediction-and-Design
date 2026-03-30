#!/usr/bin/env python3
"""
Export SHAP heatmaps from shap_results.json to PNG files.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
from pathlib import Path
import traceback

import click
import numpy as np


STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")


def allele_to_filename(allele: str) -> str:
    """Convert an allele name into a filesystem-safe stem."""
    return allele.replace("*", "_").replace(":", "_")


def load_heatmaps(json_path: Path) -> dict[str, dict[str, list[list[float]]]]:
    """Load the per-allele heatmaps from the SHAP JSON output."""
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        allele: payload.get("heatmaps", {})
        for allele, payload in data.items()
        if payload.get("heatmaps")
    }


def pick_alleles(
    available: Iterable[str],
    requested: str | None,
) -> list[str]:
    """Return the allele list to render."""
    available_list = list(available)
    if not requested:
        return available_list
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    return [allele for allele in wanted if allele in available_list]


def save_allele_heatmap(
    allele: str,
    heatmaps: dict[str, list[list[float]]],
    output_dir: Path,
) -> Path:
    """Render one PNG containing all available peptide-length heatmaps."""
    plt = importlib.import_module("matplotlib.pyplot")

    lengths = sorted(heatmaps.keys(), key=int)
    arrays = {
        length: np.asarray(heatmaps[length], dtype=np.float64)
        for length in lengths
    }
    vmax = max(float(np.abs(array).max()) for array in arrays.values())
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(
        1,
        len(lengths),
        figsize=(4.8 * len(lengths), 5.5),
        constrained_layout=True,
    )
    if len(lengths) == 1:
        axes = [axes]

    image = None
    for axis, length in zip(axes, lengths):
        array = arrays[length]
        image = axis.imshow(
            array,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        axis.set_title(f"{allele} | {length}-mer")
        axis.set_xticks(range(len(STANDARD_AAS)))
        axis.set_xticklabels(STANDARD_AAS, rotation=90)
        axis.set_yticks(range(array.shape[0]))
        axis.set_yticklabels([f"P{i + 1}" for i in range(array.shape[0])])
        axis.set_xlabel("Amino Acid")
        axis.set_ylabel("Position")

    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.85, label="Mean SHAP Value")
    fig.suptitle(f"SHAP Heatmaps for {allele}", fontsize=14)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{allele_to_filename(allele)}_heatmaps.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Export SHAP heatmaps from shap_results.json to PNG.",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(path_type=Path),
    default=Path("explainer/shap_results.json"),
    show_default=True,
    help="Path to shap_results.json",
)
@click.option(
    "--out_dir",
    type=click.Path(path_type=Path),
    default=Path("explainer/heatmaps"),
    show_default=True,
    help="Directory for exported PNG files",
)
@click.option(
    "--alleles",
    default=None,
    help="Optional comma-separated allele list",
)
def main(
    json_path: Path,
    out_dir: Path,
    alleles: str | None,
) -> None:
    """Render PNG heatmaps for one or more alleles."""
    log_path = out_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            try:
                all_heatmaps = load_heatmaps(json_path)
                selected_alleles = pick_alleles(all_heatmaps.keys(), alleles)
                if not selected_alleles:
                    raise SystemExit(
                        "No matching allele heatmaps found to export."
                    )

                for allele in selected_alleles:
                    output_path = save_allele_heatmap(
                        allele,
                        all_heatmaps[allele],
                        out_dir,
                    )
                    print(output_path)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1) from None


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
