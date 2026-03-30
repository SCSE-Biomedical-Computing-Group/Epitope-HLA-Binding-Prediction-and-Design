"""
Checkpoint loading/export helpers for generator workflows.
"""

from datetime import datetime
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Callable, Dict

import torch

from generator.logging_utils import get_run_logger


LOGGER = get_run_logger(__name__)


def identity_checkpoint_upgrade(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Return checkpoints unchanged when no legacy-upgrade helper is available.
    """
    return state_dict


def get_checkpoint_upgrade_fn() -> Callable[
    [Dict[str, torch.Tensor]],
    Dict[str, torch.Tensor],
]:
    """
    Return the legacy-upgrade helper when available, else a no-op fallback.
    """
    from generator import ar_model as ar_model_module

    return getattr(
        ar_model_module,
        "upgrade_legacy_ar_checkpoint",
        identity_checkpoint_upgrade,
    )


def export_ar_preset(
    source_ckpt: Path,
    preset_ckpt: Path,
    run_id: str,
    args: SimpleNamespace,
    train_size: int,
    val_size: int,
) -> None:
    """
    Copy the trained AR checkpoint to a stable preset path for reuse.
    """
    preset_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_ckpt, preset_ckpt)

    meta_path = preset_ckpt.with_suffix(".meta.json")
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(source_ckpt),
        "preset_checkpoint": str(preset_ckpt),
        "data": str(args.data),
        "embed_dim": args.ar_embed_dim,
        "epochs": args.ar_epochs,
        "batch_size": args.ar_batch_size,
        "learning_rate": args.ar_lr,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "train_size": train_size,
        "val_size": val_size,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info("Reusable AR preset saved to %s", preset_ckpt)
    LOGGER.info("Preset metadata saved to %s", meta_path)
    LOGGER.info("Reuse later with: --ar_ckpt %s", preset_ckpt)
