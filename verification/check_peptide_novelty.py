#!/usr/bin/env python3
"""
Check whether generated peptides appear in the human proteome.

Uses a sliding-window approach grouped by peptide length so each protein is
scanned once per length class.
"""

from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import gzip
from itertools import combinations
from pathlib import Path
import sys
import traceback

import click
import pandas as pd

AA_STD = set("ACDEFGHIKLMNPQRSTVWY")
AA_LIST = sorted(AA_STD)


def iter_fasta(fasta_path: Path):
    """Yield (seq_id, seq) from FASTA; supports .gz."""
    opener = gzip.open if fasta_path.suffix == ".gz" else open
    with opener(fasta_path, "rt") as handle:
        seq_id, chunks = None, []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(chunks)
                seq_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if seq_id is not None:
            yield seq_id, "".join(chunks)


def nearest_hamming_distance_within(
    peptide: str,
    windows: set,
    max_dist: int,
) -> int | None:
    """
    Return nearest Hamming distance in [0, max_dist] if a match exists, else None.
    Only supports max_dist up to 2 to keep runtime predictable.
    """
    if peptide in windows:
        return 0
    if max_dist <= 0:
        return None

    # Distance-1 neighbors
    chars = list(peptide)
    for i, orig in enumerate(chars):
        prefix = peptide[:i]
        suffix = peptide[i + 1:]
        for aa in AA_LIST:
            if aa == orig:
                continue
            candidate = prefix + aa + suffix
            if candidate in windows:
                return 1

    if max_dist == 1:
        return None
    if max_dist > 2:
        raise ValueError("near-match search currently supports max_dist up to 2")

    # Distance-2 neighbors
    L = len(peptide)
    for i, j in combinations(range(L), 2):
        oi = peptide[i]
        oj = peptide[j]
        for ai in AA_LIST:
            if ai == oi:
                continue
            for aj in AA_LIST:
                if aj == oj:
                    continue
                candidate = peptide[:i] + ai + peptide[i + 1:j] + aj + peptide[j + 1:]
                if candidate in windows:
                    return 2

    return None


def check_peptides_against_proteome(
    peptides,
    fasta_path: Path,
    max_hits_per_peptide=10,
    near_match_max_dist: int = 0,
):
    """
    Check a list of peptides against a proteome FASTA.

    Returns records with keys:
        peptide, num_proteins_matched, matches [(seq_id, pos)...], is_novel
    """
    peptides = sorted(set(p.strip().upper() for p in peptides if isinstance(p, str)))
    peptides = [p for p in peptides if set(p).issubset(AA_STD)]

    pep_by_len = defaultdict(set)
    for peptide in peptides:
        pep_by_len[len(peptide)].add(peptide)

    hits = {peptide: [] for peptide in peptides}
    hit_proteins = {peptide: set() for peptide in peptides}
    windows_by_len = defaultdict(set) if near_match_max_dist > 0 else None

    lengths = sorted(pep_by_len.keys())
    for seq_id, seq in iter_fasta(fasta_path):
        seq = seq.upper()
        n = len(seq)
        for length in lengths:
            if n < length:
                continue
            pepset = pep_by_len[length]
            for i in range(n - length + 1):
                window = seq[i:i + length]
                if windows_by_len is not None:
                    windows_by_len[length].add(window)
                if window in pepset:
                    if len(hits[window]) < max_hits_per_peptide:
                        hits[window].append((seq_id, i + 1))
                    hit_proteins[window].add(seq_id)

    results = []
    for peptide in peptides:
        exact_novel = len(hit_proteins[peptide]) == 0
        near_dist = None
        has_near_match = False
        strict_novel = exact_novel
        if near_match_max_dist > 0:
            near_dist = nearest_hamming_distance_within(
                peptide,
                windows_by_len[len(peptide)],
                near_match_max_dist,
            )
            # Count non-exact close neighbors as near matches
            has_near_match = near_dist is not None and near_dist > 0
            strict_novel = exact_novel and not has_near_match

        results.append({
            "peptide": peptide,
            "num_proteins_matched": len(hit_proteins[peptide]),
            "matches": hits[peptide],
            "is_novel": exact_novel,
            "has_near_match": has_near_match,
            "near_match_distance": near_dist,
            "is_strict_novel": strict_novel,
        })
    return results


def find_latest_run(run_base: Path) -> Path:
    """Return latest timestamped run directory under run_base."""
    if not run_base.exists():
        raise FileNotFoundError(f"Run base does not exist: {run_base}")
    candidates = sorted(
        [p for p in run_base.iterdir() if p.is_dir() and p.name.startswith("20")],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No timestamped run directories found under {run_base}")
    return candidates[0]


def resolve_proteome(user_path: Path | None, here: Path, root: Path) -> Path:
    """Resolve proteome path from CLI override or common defaults."""
    candidates = []
    if user_path is not None:
        candidates.append(user_path)
    candidates.extend([
        here / "human_proteome.fasta",
        here / "human_proteome.fasta.gz",
        root / "data" / "human_proteome.fasta",
        Path("human_proteome.fasta"),
        Path("UP000005640_9606.fasta"),
    ])
    fasta = next((p for p in candidates if p.exists()), None)
    if fasta is None:
        raise FileNotFoundError(
            "Human proteome FASTA not found. Download with:\n"
            "curl -o verification/human_proteome.fasta.gz "
            "'https://rest.uniprot.org/uniprotkb/stream?compressed=true"
            "&format=fasta&query=%28proteome%3AUP000005640%29'"
        )
    return fasta


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Check generated peptide novelty vs human proteome",
)
@click.option(
    "--run_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to a specific generation run directory (contains allele subdirs)",
)
@click.option(
    "--run_base",
    type=click.Path(path_type=Path),
    default=Path("outputs/ms_run1"),
    show_default=True,
    help="Base directory used when --run_dir is not provided (latest run is used)",
)
@click.option(
    "--csv_name",
    default="final_peptides_div.csv",
    show_default=True,
    help="CSV name to load under each allele directory",
)
@click.option(
    "--proteome",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to human proteome FASTA(.gz)",
)
@click.option(
    "--max_hits_per_peptide",
    type=int,
    default=10,
    show_default=True,
    help="Max number of match positions stored per peptide",
)
@click.option(
    "--near_match_max_dist",
    type=int,
    default=1,
    show_default=True,
    help="Also flag non-exact near matches within this Hamming distance (0 disables, max=2)",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("verification/peptide_human_proteome_hits.csv"),
    show_default=True,
    help="Output CSV path",
)
def main(
    run_dir: Path | None,
    run_base: Path,
    csv_name: str,
    proteome: Path | None,
    max_hits_per_peptide: int,
    near_match_max_dist: int,
    out: Path,
) -> None:
    log_path = out.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            try:
                _run_novelty_check(
                    run_dir=run_dir,
                    run_base=run_base,
                    csv_name=csv_name,
                    proteome=proteome,
                    max_hits_per_peptide=max_hits_per_peptide,
                    near_match_max_dist=near_match_max_dist,
                    out=out,
                )
            except Exception:
                traceback.print_exc()
                raise SystemExit(1) from None


def _run_novelty_check(
    *,
    run_dir: Path | None,
    run_base: Path,
    csv_name: str,
    proteome: Path | None,
    max_hits_per_peptide: int,
    near_match_max_dist: int,
    out: Path,
) -> None:
    """Execute the proteome novelty check workflow."""
    here = Path(__file__).parent
    root = here.parent

    try:
        fasta = resolve_proteome(proteome, here, root)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    print(f"✅ Proteome: {fasta}")

    resolved_run_dir = run_dir or find_latest_run(run_base)
    print(f"✅ Run directory: {resolved_run_dir}")
    csv_files = sorted(resolved_run_dir.glob(f"*/{csv_name}"))
    if not csv_files:
        print(f"❌ No result CSVs named {csv_name} under {resolved_run_dir}")
        sys.exit(1)

    frames = []
    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        frames.append(frame)
        print(f"  Loaded {len(frame)} peptides from {csv_path.parent.name}")

    all_df = pd.concat(frames, ignore_index=True)
    peptides = all_df["peptide"].dropna().tolist()
    print(f"\nTotal unique peptides to check: {len(set(peptides))}")

    print("Scanning proteome (this may take a minute)...")
    results = check_peptides_against_proteome(
        peptides,
        fasta,
        max_hits_per_peptide=max_hits_per_peptide,
        near_match_max_dist=near_match_max_dist,
    )

    novelty_df = pd.DataFrame(
        [
            {
                "peptide": r["peptide"],
                "is_novel": r["is_novel"],
                "num_proteins_matched": r["num_proteins_matched"],
                "first_match_protein": r["matches"][0][0] if r["matches"] else None,
                "first_match_pos": r["matches"][0][1] if r["matches"] else None,
                "has_near_match": r["has_near_match"],
                "near_match_distance": r["near_match_distance"],
                "is_strict_novel": r["is_strict_novel"],
            }
            for r in results
        ]
    )

    merged = all_df.merge(novelty_df, on="peptide", how="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    novel_count = int(novelty_df["is_novel"].sum())
    strict_novel_count = int(novelty_df["is_strict_novel"].sum())
    near_count = int(novelty_df["has_near_match"].sum())
    total = len(novelty_df)
    print("\nNovelty Summary")
    print(f"  Total unique peptides  : {total}")
    print(f"  Exact novel            : {novel_count}  ({100 * novel_count / total:.1f}%)")
    print(f"  Self-antigen hits      : {total - novel_count}  ({100 * (total - novel_count) / total:.1f}%)")
    if near_match_max_dist > 0:
        print(
            f"  Near matches (dist≤{near_match_max_dist}): "
            f"{near_count}  ({100 * near_count / total:.1f}%)"
        )
        print(f"  Strict novel           : {strict_novel_count}  ({100 * strict_novel_count / total:.1f}%)")
    print("\nPer-allele breakdown:")
    for allele, group in merged.groupby("allele"):
        allele_novel = int(group["is_novel"].sum())
        if near_match_max_dist > 0:
            allele_strict = int(group["is_strict_novel"].sum())
            print(
                f"  {allele}: exact={allele_novel}/{len(group)} ({100 * allele_novel / len(group):.0f}%), "
                f"strict={allele_strict}/{len(group)} ({100 * allele_strict / len(group):.0f}%)"
            )
        else:
            print(f"  {allele}: {allele_novel}/{len(group)} novel ({100 * allele_novel / len(group):.0f}%)")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
