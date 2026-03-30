"""
Training and sampling utilities for the autoregressive peptide generator.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Protocol, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from generator.ar_model import AA_TO_IDX, PAD_IDX, VOCAB_SIZE, encode_peptide
from generator.logging_utils import get_run_logger


PeptidePairs = Tuple[List[str], List[str]]
LOGGER = get_run_logger(__name__)


class ARGenerator(Protocol):
    """Interface needed by sampling/generation helpers."""

    def generate(
        self,
        alleles: List[str],
        lengths: List[int],
        device: torch.device,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> Tuple[List[str], torch.Tensor]:
        ...


class PeptideDataset(Dataset):
    """
    Dataset for training AR model on (allele, peptide) pairs.
    """

    def __init__(
        self,
        alleles: List[str],
        peptides: List[str],
        max_len: int = 11,
    ):
        self.alleles = alleles
        self.peptides = peptides
        self.max_len = max_len

        self.encoded = []
        self.lengths = []
        for pep in peptides:
            tokens = encode_peptide(pep)
            self.lengths.append(len(pep))
            while len(tokens) < max_len + 2:
                tokens.append(PAD_IDX)
            self.encoded.append(tokens[: max_len + 2])

    def __len__(self) -> int:
        return len(self.peptides)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "allele": self.alleles[idx],
            "tokens": torch.tensor(self.encoded[idx], dtype=torch.long),
            "length": self.lengths[idx],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    alleles = [item["allele"] for item in batch]
    tokens = torch.stack([item["tokens"] for item in batch])
    lengths = torch.tensor(
        [item["length"] for item in batch],
        dtype=torch.long,
    )
    return {"alleles": alleles, "tokens": tokens, "lengths": lengths}


class TrainingHistory(TypedDict):
    """Structured return type for AR training metrics."""

    train_loss: List[float]
    val_loss: List[float]
    train_ppl: List[float]
    val_ppl: List[float]
    best_epoch: int
    epochs_ran: int


def train_ar_model(
    model: nn.Module,
    train_data: PeptidePairs,
    val_data: Optional[PeptidePairs] = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[Path] = None,
    verbose: bool = True,
    early_stopping_patience: Optional[int] = 8,
    early_stopping_min_delta: float = 1e-4,
    min_epochs_before_stop: int = 5,
) -> TrainingHistory:
    """
    Train the AR model on peptide data.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    train_dataset = PeptideDataset(train_data[0], train_data[1])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    if val_data:
        val_dataset = PeptideDataset(val_data[0], val_data[1])
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    history: TrainingHistory = {
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
        "best_epoch": 0,
        "epochs_ran": 0,
    }
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            disable=not verbose or not sys.stderr.isatty(),
        )
        for batch in pbar:
            tokens = batch["tokens"].to(device)
            lengths = batch["lengths"].to(device)
            alleles = batch["alleles"]

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            logits = model(input_tokens, alleles, lengths, device)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                target_tokens.reshape(-1),
                ignore_index=PAD_IDX,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
            pbar.set_postfix({"loss": float(loss.item())})

        avg_train_loss = total_loss / max(num_batches, 1)
        history["train_loss"].append(avg_train_loss)
        history["train_ppl"].append(float(np.exp(avg_train_loss)))

        if val_data:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    tokens = batch["tokens"].to(device)
                    lengths = batch["lengths"].to(device)
                    alleles = batch["alleles"]

                    input_tokens = tokens[:, :-1]
                    target_tokens = tokens[:, 1:]

                    logits = model(input_tokens, alleles, lengths, device)
                    loss = F.cross_entropy(
                        logits.view(-1, VOCAB_SIZE),
                        target_tokens.reshape(-1),
                        ignore_index=PAD_IDX,
                    )
                    val_loss += float(loss.item())
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            avg_val_ppl = float(np.exp(avg_val_loss))
            history["val_loss"].append(avg_val_loss)
            history["val_ppl"].append(avg_val_ppl)

            improved = avg_val_loss < (
                best_val_loss - early_stopping_min_delta
            )
            if improved:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improve = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
            else:
                epochs_without_improve += 1

            if verbose:
                LOGGER.info(
                    "Epoch %s: train_loss=%.4f, val_loss=%.4f, val_ppl=%.2f",
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    avg_val_ppl,
                )
        else:
            if verbose:
                LOGGER.info(
                    "Epoch %s: train_loss=%.4f",
                    epoch + 1,
                    avg_train_loss,
                )
            if save_path and (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), save_path)

        scheduler.step()

        if (
            val_data
            and early_stopping_patience is not None
            and early_stopping_patience > 0
            and (epoch + 1) >= max(1, min_epochs_before_stop)
            and epochs_without_improve >= early_stopping_patience
        ):
            if verbose:
                LOGGER.info(
                    "Early stopping at epoch %s: no validation improvement "
                    "for %s epochs (best epoch=%s, best_val_loss=%.4f).",
                    epoch + 1,
                    epochs_without_improve,
                    best_epoch,
                    best_val_loss,
                )
            break

    if save_path and not val_data:
        torch.save(model.state_dict(), save_path)
    elif val_data and save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))

    if history["val_ppl"]:
        best_ppl = min(history["val_ppl"])
        LOGGER.info("Best validation perplexity: %.2f", best_ppl)
        if best_epoch > 0:
            LOGGER.info("Best validation epoch: %s", best_epoch)
    elif history["train_ppl"]:
        LOGGER.info("Final train perplexity: %.2f", history["train_ppl"][-1])

    history["best_epoch"] = best_epoch
    history["epochs_ran"] = len(history["train_loss"])
    return history


def stratified_train_val_split(
    alleles: List[str],
    peptides: List[str],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[PeptidePairs, PeptidePairs]:
    """
    Split data into train/val sets stratified by (allele, peptide length).
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    n = len(peptides)
    if n < 2:
        return (alleles, peptides), ([], [])

    groups: Dict[Tuple[str, int], List[int]] = {}
    for idx, (allele, peptide) in enumerate(zip(alleles, peptides)):
        groups.setdefault((allele, len(peptide)), []).append(idx)

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []

    for indices in groups.values():
        idxs = np.array(indices, dtype=int)
        rng.shuffle(idxs)
        if len(idxs) == 1:
            train_idx.extend(idxs.tolist())
            continue
        n_val = int(round(len(idxs) * val_fraction))
        n_val = max(1, n_val)
        n_val = min(n_val, len(idxs) - 1)
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    if not train_idx or not val_idx:
        all_idx = np.arange(n, dtype=int)
        rng.shuffle(all_idx)
        n_val = max(1, int(round(n * val_fraction)))
        n_val = min(n_val, n - 1)
        val_idx = all_idx[:n_val].tolist()
        train_idx = all_idx[n_val:].tolist()

    train_data = (
        [alleles[i] for i in train_idx],
        [peptides[i] for i in train_idx],
    )
    val_data = (
        [alleles[i] for i in val_idx],
        [peptides[i] for i in val_idx],
    )
    return train_data, val_data


def sample_diverse_candidates(
    model: ARGenerator,
    allele: str,
    target_count: int,
    length_distribution: Dict[int, float],
    device: torch.device,
    temperature_schedule: List[float] = [0.7, 0.9, 1.1, 1.3],
    top_p_schedule: List[float] = [0.85, 0.9, 0.95],
    batch_size: int = 64,
    max_attempts: int = 10,
    verbose: bool = True,
) -> List[Dict]:
    """
    Generate diverse candidate peptides for a single allele.
    Uses multiple temperature/top_p settings to encourage diversity.
    """
    all_peptides = set()
    results = []
    samples_per_length = {
        length: int(target_count * proportion * 2)
        for length, proportion in length_distribution.items()
    }

    for attempt in range(max_attempts):
        if len(all_peptides) >= target_count:
            break

        temp = temperature_schedule[attempt % len(temperature_schedule)]
        top_p = top_p_schedule[attempt % len(top_p_schedule)]

        for length, num_samples in samples_per_length.items():
            if num_samples == 0:
                continue

            batch_alleles = [allele] * min(num_samples, batch_size)
            batch_lengths = [length] * len(batch_alleles)

            with torch.no_grad():
                peptides, log_probs = model.generate(
                    batch_alleles,
                    batch_lengths,
                    device,
                    temperature=temp,
                    top_p=top_p,
                )

            for peptide, log_prob in zip(peptides, log_probs):
                if len(peptide) == length and peptide not in all_peptides:
                    if all(aa in AA_TO_IDX for aa in peptide):
                        all_peptides.add(peptide)
                        results.append(
                            {
                                "allele": allele,
                                "peptide": peptide,
                                "length": length,
                                "log_prob": float(log_prob.item()),
                                "source": "AR",
                                "temperature": temp,
                                "top_p": top_p,
                            }
                        )

    if verbose:
        LOGGER.info(
            "Generated %s unique peptides for %s",
            len(results),
            allele,
        )

    return results
