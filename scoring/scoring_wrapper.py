"""
NetMHCpan-backed scoring wrapper for generator workflows.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from generator.logging_utils import get_run_logger


LOGGER = get_run_logger(__name__)


def _resolve_netmhcpan(
    netmhcpan_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Resolve the NetMHCpan binary and home directory."""
    if netmhcpan_path and Path(netmhcpan_path).exists():
        home = Path(netmhcpan_path)
    elif os.environ.get("NETMHCPAN"):
        home = Path(os.environ["NETMHCPAN"])
    else:
        project_root = Path(__file__).resolve().parents[2]
        home = project_root.parent / "netMHCpan-4.2"

    if not home.exists():
        raise FileNotFoundError(
            f"NetMHCpan not found at {home}. "
            "Set NETMHCPAN env var or pass --netmhcpan."
        )

    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        binary = home / "Darwin_arm64" / "bin" / "netMHCpan-4.2"
    elif system == "Darwin":
        binary = home / "Darwin_x86_64" / "bin" / "netMHCpan-4.2"
    else:
        binary = home / "Linux_x86_64" / "bin" / "netMHCpan-4.2"

    if not binary.exists():
        raise FileNotFoundError(f"NetMHCpan binary not found at {binary}")

    return binary, home


def _run_netmhcpan(
    peptides: List[str],
    allele: str,
    binary: Path,
    home: Path,
    timeout: int = 120,
) -> Dict[str, float]:
    """Run NetMHCpan for one allele and return peptide->affinity(nM)."""
    netmhc_allele = allele.replace("*", "")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
    ) as handle:
        for peptide in peptides:
            handle.write(f"{peptide}\n")
        temp_path = Path(handle.name)

    env = os.environ.copy()
    env["NMHOME"] = str(home)
    env["NETMHCpan"] = str(home)
    env["TMPDIR"] = tempfile.gettempdir()

    try:
        result = subprocess.run(
            [str(binary), "-p", str(temp_path), "-a", netmhc_allele, "-BA"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(home),
            check=False,
        )
        affinities: Dict[str, float] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 16 and parts[0].isdigit():
                try:
                    peptide = parts[2]
                    affinity_nm = float(parts[15])
                    affinities[peptide] = affinity_nm
                except ValueError:
                    continue
        return affinities
    except subprocess.TimeoutExpired:
        return {}
    finally:
        temp_path.unlink(missing_ok=True)


class PredictorWrapper:
    """
    Wrapper that scores peptide-allele pairs via NetMHCpan 4.2.

    The public API (.score / __call__) is stable for generator/refinement code.
    """

    def __init__(
        self,
        netmhcpan_path: Optional[Path] = None,
        device: Optional[object] = None,  # kept for API compatibility
        data_path: Optional[Path] = None,  # kept for API compatibility
        batch_size: int = 2000,
        netmhcpan_jobs: Optional[int] = None,
    ):
        _ = device
        _ = data_path

        self._binary, self._home = _resolve_netmhcpan(netmhcpan_path)
        self._batch_size = batch_size
        jobs_from_env = os.environ.get("NETMHCPAN_JOBS")
        if netmhcpan_jobs is None and jobs_from_env is not None:
            try:
                netmhcpan_jobs = int(jobs_from_env)
            except ValueError:
                netmhcpan_jobs = 1
        self._netmhcpan_jobs = max(1, int(netmhcpan_jobs or 1))
        LOGGER.info("PredictorWrapper: using NetMHCpan at %s", self._binary)
        LOGGER.info(
            "PredictorWrapper: NetMHCpan parallel jobs=%s, batch_size=%s",
            self._netmhcpan_jobs,
            self._batch_size,
        )

    def score(self, alleles: List[str], peptides: List[str]) -> np.ndarray:
        """
        Score peptide-allele pairs via NetMHCpan.
        """
        scores = np.zeros(len(peptides))

        # Group by allele to batch NetMHCpan calls.
        allele_groups: Dict[str, List[int]] = {}
        for i, allele in enumerate(alleles):
            allele_groups.setdefault(allele, []).append(i)

        batch_tasks = []
        for allele, indices in allele_groups.items():
            peps = [peptides[i] for i in indices]
            for start in range(0, len(peps), self._batch_size):
                batch = peps[start:start + self._batch_size]
                batch_tasks.append((allele, indices, start, batch))

        def _apply_batch_scores(
            allele: str,
            indices: List[int],
            start: int,
            batch: List[str],
            affinity_map: Dict[str, float],
        ) -> None:
            _ = allele
            for offset, peptide in enumerate(batch):
                affinity_nm = affinity_map.get(peptide, 50000.0)
                scores[indices[start + offset]] = -np.log10(max(affinity_nm, 1e-9))

        if self._netmhcpan_jobs <= 1 or len(batch_tasks) <= 1:
            for allele, indices, start, batch in batch_tasks:
                affinity_map = _run_netmhcpan(
                    batch,
                    allele,
                    binary=self._binary,
                    home=self._home,
                )
                _apply_batch_scores(allele, indices, start, batch, affinity_map)
            return scores

        with ThreadPoolExecutor(max_workers=self._netmhcpan_jobs) as executor:
            future_map = {
                executor.submit(
                    _run_netmhcpan,
                    batch,
                    allele,
                    self._binary,
                    self._home,
                ): (allele, indices, start, batch)
                for allele, indices, start, batch in batch_tasks
            }
            for future in as_completed(future_map):
                allele, indices, start, batch = future_map[future]
                try:
                    affinity_map = future.result()
                except Exception:
                    affinity_map = {}
                _apply_batch_scores(allele, indices, start, batch, affinity_map)

        return scores

    def __call__(self, alleles: List[str], peptides: List[str]) -> np.ndarray:
        """Alias for score()."""
        return self.score(alleles, peptides)
