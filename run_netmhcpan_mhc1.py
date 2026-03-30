#!/usr/bin/env python3
"""
Run NetMHCpan 4.2 on MHC Dataset
Evaluates NetMHCpan predictions and computes performance metrics

Usage:
    python run_netmhcpan_mhc1.py [OPTIONS]

Options:
    --data PATH         Input CSV file (default: data/mhc1.csv)
    --output-dir PATH   Output directory (default: results/)
    --netmhcpan PATH    Path to NetMHCpan installation (default: $NETMHCPAN or ~/Downloads/netMHCpan-4.2)
    --sample-size N     Number of samples to evaluate (default: 10000, 0 = all)
    --threshold N       Binding threshold in nM (default: 500)
"""

import sys
import os
import subprocess
import tempfile as _tempfile
from types import SimpleNamespace
import atexit

import click
import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def convert_allele_name(allele_name):
    """
    Convert allele name from dataset format to NetMHCpan format.
    Dataset: HLA-A*02:01
    NetMHCpan: HLA-A02:01 (without asterisk)
    """
    return allele_name.replace('*', '')

# ============================================================================
# CLICK ARGUMENT PARSING
# ============================================================================
def parse_args(argv=None):
    @click.command(
        context_settings={"help_option_names": ["-h", "--help"]},
        help="Run NetMHCpan 4.2 on MHC dataset",
    )
    @click.option(
        "--data",
        type=click.Path(path_type=Path),
        default=Path("data/mhc1.csv"),
        show_default=True,
        help="Input CSV file",
    )
    @click.option(
        "--output-dir",
        "output_dir",
        type=click.Path(path_type=Path),
        default=Path("results"),
        show_default=True,
        help="Output directory",
    )
    @click.option(
        "--netmhcpan",
        type=click.Path(path_type=Path),
        default=None,
        help="Path to NetMHCpan installation",
    )
    @click.option(
        "--sample-size",
        "sample_size",
        type=int,
        default=10000,
        show_default=True,
        help="Number of samples (0 = all)",
    )
    @click.option(
        "--threshold",
        type=float,
        default=500.0,
        show_default=True,
        help="Binding threshold in nM",
    )
    def _cli(
        data: Path,
        output_dir: Path,
        netmhcpan: Path | None,
        sample_size: int,
        threshold: float,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            data=data,
            output_dir=output_dir,
            netmhcpan=netmhcpan,
            sample_size=sample_size,
            threshold=threshold,
        )

    try:
        result = _cli.main(
            args=argv,
            prog_name="run_netmhcpan_mhc1.py",
            standalone_mode=False,
        )
        if isinstance(result, SimpleNamespace):
            return result
        raise SystemExit(int(result or 0))
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from None


args = parse_args(sys.argv[1:])
args.output_dir.mkdir(parents=True, exist_ok=True)
_log_handle = (args.output_dir / "run.log").open("a", encoding="utf-8")


def _log(*args, sep: str = " ", end: str = "\n") -> None:
    """Write log lines to the run log file."""
    _log_handle.write(sep.join(str(arg) for arg in args) + end)
    _log_handle.flush()


sys.stdout = _log_handle
sys.stderr = _log_handle
atexit.register(_log_handle.close)

_log("="*80)
_log("NETMHCPAN 4.2 - COMBINED MHC1 DATASET EVALUATION")
_log("="*80)

# Change to script directory's parent (project root)
script_dir = Path(__file__).resolve().parent.parent
os.chdir(script_dir)
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
_log(f"Working directory: {os.getcwd()}")

# ============================================================================
# STEP 1: Setup NetMHCpan
# ============================================================================
_log("\nStep 1: Setting up NetMHCpan...")

# Determine NetMHCpan path: CLI arg > env var > sibling directory
if args.netmhcpan:
    netmhc_home = args.netmhcpan
elif os.environ.get('NETMHCPAN'):
    netmhc_home = Path(os.environ['NETMHCPAN'])
else:
    # Try sibling directory relative to project root
    netmhc_home = script_dir.parent / 'netMHCpan-4.2'
    if not netmhc_home.exists():
        _log("[ERROR] NetMHCpan not found. Set NETMHCPAN env var or pass --netmhcpan.")
        sys.exit(1)

# Detect platform-specific binary
import platform
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    netmhc_bin = netmhc_home / 'Darwin_arm64' / 'bin' / 'netMHCpan-4.2'
elif platform.system() == 'Darwin':
    netmhc_bin = netmhc_home / 'Darwin_x86_64' / 'bin' / 'netMHCpan-4.2'
else:
    netmhc_bin = netmhc_home / 'Linux_x86_64' / 'bin' / 'netMHCpan-4.2'

if not netmhc_bin.exists():
    _log(f"[ERROR] NetMHCpan not found at {netmhc_bin}")
    sys.exit(1)

_log(f"   [OK] Found: {netmhc_bin}")

# Set environment variables
os.environ['NMHOME'] = str(netmhc_home)
os.environ['NETMHCpan'] = str(netmhc_home)
os.environ['TMPDIR'] = _tempfile.gettempdir()

_log(f"   Set NETMHCpan={netmhc_home}")

# ============================================================================
# STEP 2: Load Dataset
# ============================================================================
_log("\n" + "="*80)
_log("LOADING COMBINED MHC1 DATASET")
_log("="*80)

data_path = args.data
if not data_path.exists():
    _log(f"\n[ERROR] Data not found: {data_path}")
    sys.exit(1)

_log(f"\nLoading {data_path}...")
data = pd.read_csv(data_path)
_log(f"   [OK] Loaded {len(data):,} samples")
_log(f"   Unique alleles: {data['allele'].nunique():,}")
_log(f"   Unique peptides: {data['peptide'].nunique():,}")

# ============================================================================
# STEP 3: Filter and Sample Data
# ============================================================================
_log("\nFiltering data...")

# Filter for peptide length (8-11 amino acids)
data = data[
    (data['peptide'].str.len() >= 8) & 
    (data['peptide'].str.len() <= 11)
].copy()
_log(f"   After length filter (8-11 aa): {len(data):,} samples")

# Remove non-standard amino acids
standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
data = data[
    data['peptide'].apply(lambda p: all(aa in standard_aa for aa in p))
].copy()
_log(f"   After AA filter: {len(data):,} samples")

# Sample for manageable runtime
SAMPLE_SIZE = args.sample_size

if SAMPLE_SIZE > 0 and len(data) > SAMPLE_SIZE:
    _log(f"\nSampling {SAMPLE_SIZE:,} for evaluation...")
    _log(f"   (Full dataset would take ~{len(data)/5000*8:.0f} hours)")
    data = data.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
else:
    _log(f"\n[OK] Using all {len(data):,} samples")

_log("\nFinal evaluation dataset:")
_log(f"   Samples: {len(data):,}")
_log(f"   Unique alleles: {data['allele'].nunique():,}")
_log(f"   Unique peptides: {data['peptide'].nunique():,}")

# Show HLA distribution
_log("\nHLA Alpha Distribution:")
if 'hla_alpha' in data.columns:
    for alpha, count in data['hla_alpha'].value_counts().items():
        _log(f"   - HLA-{alpha}: {count:,} ({count/len(data)*100:.1f}%)")

# ============================================================================
# STEP 4: Run NetMHCpan Predictions
# ============================================================================
_log("\n" + "="*80)
_log("RUNNING NETMHCPAN PREDICTIONS")
_log("="*80)
_log(f"\nEstimated time: {len(data['allele'].unique())*2/60:.1f} minutes\n")

predictions = []
unique_alleles = data['allele'].unique()
total_alleles = len(unique_alleles)

start_time = time.time()

for idx, allele in enumerate(unique_alleles, 1):
    # Progress updates
    if idx % 10 == 0 or idx == 1:
        elapsed = time.time() - start_time
        rate = idx / elapsed if elapsed > 0 else 0
        remaining = (total_alleles - idx) / rate if rate > 0 else 0
        _log(f"   [{idx}/{total_alleles}] {idx/total_alleles*100:.1f}% | "
              f"Time: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m | "
              f"Predictions: {len(predictions):,}")
    
    # Get peptides for this allele
    allele_data = data[data['allele'] == allele]
    peptides = allele_data['peptide'].unique()
    
    # Convert allele name to NetMHCpan format
    netmhc_allele = convert_allele_name(allele)
    
    # Create temp file with peptides
    with _tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for pep in peptides:
            f.write(f"{pep}\n")
        temp_file = f.name
    
    try:
        # Run NetMHCpan
        env = os.environ.copy()
        env['NMHOME'] = str(netmhc_home)
        env['NETMHCpan'] = str(netmhc_home)
        env['TMPDIR'] = _tempfile.gettempdir()
        
        result = subprocess.run(
            [str(netmhc_bin), '-p', temp_file, '-a', netmhc_allele, '-BA'],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
            cwd=str(netmhc_home),
            check=False,
        )
        
        # Debug output for first allele
        if idx == 1:
            _log(f"\n[WARN] NetMHCpan stderr: {result.stderr[:500]}")
            _log(f"[WARN] NetMHCpan stdout length: {len(result.stdout)} chars")
            _log(f"[WARN] NetMHCpan stdout preview: {result.stdout[:500]}")
            _log(f"[WARN] NetMHCpan stdout tail: {result.stdout[-500:]}")
        
        # Parse NetMHCpan output
        lines = result.stdout.split('\n')
        
        # Find header line
        header_idx = None
        for i, line in enumerate(lines):
            if 'Pos' in line and 'Peptide' in line and 'Aff(nM)' in line:
                header_idx = i
                break
        
        # Debug parsing
        if idx == 1:
            _log(f"\nDEBUG parsing for {allele}:")
            _log(f"   Header found at line: {header_idx}")
            if header_idx is not None:
                _log(f"   Total lines: {len(lines)}")
                _log(f"   Parsing from line: {header_idx+2}")
        
        # Parse data lines
        if header_idx is not None:
            for line in lines[header_idx+2:]:
                if not line.strip() or line.strip().startswith('#'):
                    continue
                parts = line.strip().split()
                
                # Debug first line
                if idx == 1 and len(parts) >= 16 and parts[0].isdigit():
                    _log(f"   First data line parts count: {len(parts)}")
                    _log(f"   First data line: {line.strip()[:200]}")
                    _log(f"   Peptide (parts[2]): {parts[2]}")
                    _log(f"   Score_BA (parts[12]): {parts[12]}")
                    _log(f"   %Rank_BA (parts[13]): {parts[13]}")
                    _log(f"   Aff (parts[15]): {parts[15]}")
                    if len(parts) > 16:
                        _log(f"   BindLevel (parts[16]): {parts[16]}")
                
                if len(parts) >= 16 and parts[0].isdigit():
                    try:
                        pep = parts[2]
                        score_ba = float(parts[12])
                        rank_ba = float(parts[13])
                        aff = float(parts[15])
                        bind_level = parts[16] if len(parts) > 16 else ''
                        
                        predictions.append({
                            'peptide': pep,
                            'allele': allele,
                            'predicted_affinity': aff,
                            'score_ba': score_ba,
                            'rank_ba': rank_ba,
                            'bind_level': bind_level
                        })
                    except (ValueError, IndexError):
                        if idx == 1:
                            _log(
                                f"   [WARN] Failed to parse line: "
                                f"{line.strip()[:100]}"
                            )
    
    except subprocess.TimeoutExpired:
        pass
    except (subprocess.SubprocessError, OSError):
        pass
    finally:
        try:
            os.unlink(temp_file)
        except OSError:
            pass

total_time = time.time() - start_time

_log("\n" + "="*80)
_log(f"[OK] PREDICTIONS COMPLETE in {total_time/60:.1f} minutes")
_log("="*80)
_log(f"   Predicted: {len(predictions):,} peptide-allele pairs")

# Check if we got predictions
if len(predictions) == 0:
    _log("\n[ERROR] NetMHCpan produced 0 predictions!")
    _log("This usually means:")
    _log("  1. NetMHCpan binary is not working correctly")
    _log("  2. Data format issue (allele names not recognized)")
    _log("  3. Missing NetMHCpan data files")
    _log("\nTrying a test run...")
    
    test_peptide = "SIINFEKL"
    test_allele = "HLA-A02:01"
    with _tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(f"{test_peptide}\n")
        test_file = f.name
    
    env = os.environ.copy()
    env['NMHOME'] = str(netmhc_home)
    env['NETMHCpan'] = str(netmhc_home)
    
    test_result = subprocess.run(
        [str(netmhc_bin), '-p', test_file, '-a', test_allele, '-BA'],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    
    _log(f"\nTest command: {netmhc_bin} -p {test_file} -a {test_allele} -BA")
    _log(f"Return code: {test_result.returncode}")
    _log(f"STDOUT:\n{test_result.stdout[:1000]}")
    _log(f"STDERR:\n{test_result.stderr[:1000]}")
    
    os.unlink(test_file)
    sys.exit(1)

# ============================================================================
# STEP 5: Merge Predictions with Actual Data
# ============================================================================
_log("\nMerging predictions with actual data...")

# Convert predictions to DataFrame
pred_df = pd.DataFrame(predictions)

# Merge with original data
results = data.merge(
    pred_df,
    on=['peptide', 'allele'],
    how='inner'
)

_log(f"   Merged: {len(results):,} samples with predictions")
_log(f"   Coverage: {len(results)/len(data)*100:.1f}%")

# ============================================================================
# STEP 6: Direct NetMHCpan + SHAP Explainability
# ============================================================================
_log("\n" + "="*80)
_log("DIRECT NETMHCPAN + SHAP EXPLAINABILITY")
_log("="*80)

from explainer.shap import (
    STANDARD_AAS,
    aggregate_shap_heatmap as _aggregate_shap_heatmap,
    build_background_set as _build_background_set,
    merge_position_weights as _merge_position_weights,
    run_direct_netmhcpan_shap_bucket as _run_direct_netmhcpan_shap_bucket,
    select_foreground_set as _select_foreground_set,
)

shap_results_by_allele = {}  # allele -> {length -> heatmap (L, 20)}
shap_importance_by_allele = {}  # allele -> merged position weights (11,)

shap_data = results[
    np.isfinite(results['predicted_affinity']) &
    (results['predicted_affinity'] > 0)
].copy()

PEP_LENGTHS = [8, 9, 10, 11]
MAX_POSITIONS = 11
SHAP_BG_SIZE = int(os.environ.get("SHAP_BG_SIZE", "64"))
SHAP_FG_SIZE = int(os.environ.get("SHAP_FG_SIZE", "128"))
SHAP_NSAMPLES = int(os.environ.get("SHAP_NSAMPLES", "32"))
SHAP_MAX_ALLELES = int(os.environ.get("SHAP_MAX_ALLELES", "10"))
SHAP_NETMHCPAN_JOBS = int(os.environ.get("NETMHCPAN_JOBS", "1"))
alleles_to_explain = (
    shap_data['allele'].value_counts().head(SHAP_MAX_ALLELES).index.tolist()
)

_log(f"\n   Alleles to explain: {alleles_to_explain}")
_log(f"   Method: Direct NetMHCpan KernelSHAP (nsamples={SHAP_NSAMPLES})")
_log(
    "   SHAP budgets: "
    f"bg={SHAP_BG_SIZE}, fg={SHAP_FG_SIZE}, "
    f"max_alleles={SHAP_MAX_ALLELES}, "
    f"netmhcpan_jobs={SHAP_NETMHCPAN_JOBS}"
)

try:
    import shap

    _ = shap.__version__

    for allele in alleles_to_explain:
        a_data = shap_data[shap_data['allele'] == allele]
        heatmaps_per_len = {}
        weights_per_len = {}

        for pep_len in PEP_LENGTHS:
            bucket = a_data[a_data['peptide'].str.len() == pep_len].drop_duplicates('peptide')
            pool = bucket['peptide'].tolist()
            if len(pool) < 30:
                _log(f"   {allele} | {pep_len}-mers: skipping ({len(pool)} samples)")
                continue

            bg = _build_background_set(
                pool,
                allele,
                pep_len,
                n_background=min(SHAP_BG_SIZE, len(pool)),
                netmhcpan_path=netmhc_home,
                netmhcpan_jobs=SHAP_NETMHCPAN_JOBS,
            )
            fg = _select_foreground_set(
                pool,
                allele,
                pep_len,
                n_foreground=min(SHAP_FG_SIZE, len(pool)),
                netmhcpan_path=netmhc_home,
                netmhcpan_jobs=SHAP_NETMHCPAN_JOBS,
            )

            _log(
                f"   {allele} | {pep_len}-mers | pool={len(pool)} "
                f"| bg={len(bg)} fg={len(fg)}"
            )

            shap_3d = _run_direct_netmhcpan_shap_bucket(
                allele,
                pep_len,
                bg,
                fg,
                netmhcpan_path=netmhc_home,
                nsamples=SHAP_NSAMPLES,
                netmhcpan_jobs=SHAP_NETMHCPAN_JOBS,
            )
            heatmap, pos_importance = _aggregate_shap_heatmap(shap_3d, fg)

            heatmaps_per_len[pep_len] = heatmap
            weights_per_len[pep_len] = pos_importance

            top3 = np.argsort(-pos_importance)[:3]
            _log(
                "         Top positions: "
                + ", ".join(f"P{p+1}={pos_importance[p]:.4f}" for p in top3)
            )

        if not heatmaps_per_len:
            continue

        merged = _merge_position_weights(weights_per_len, max_positions=MAX_POSITIONS)
        shap_results_by_allele[allele] = {
            str(L): h.tolist() for L, h in heatmaps_per_len.items()
        }
        shap_importance_by_allele[allele] = merged.tolist()

    # Save SHAP outputs
    shap_out_dir = args.output_dir
    shap_out_dir.mkdir(parents=True, exist_ok=True)

    import json as _json
    shap_json_path = shap_out_dir / "shap_results.json"
    with open(shap_json_path, "w", encoding="utf-8") as _f:
        _json.dump(
            {
                "heatmaps": shap_results_by_allele,
                "position_importance": shap_importance_by_allele,
                "amino_acids": STANDARD_AAS,
                "shap_method": "kernel_shap_direct_netmhcpan",
                "xai_method": "kernel_shap_direct_netmhcpan",
                "nsamples": SHAP_NSAMPLES,
                "n_background": SHAP_BG_SIZE,
                "n_foreground": SHAP_FG_SIZE,
                "max_alleles": SHAP_MAX_ALLELES,
                "netmhcpan_jobs": SHAP_NETMHCPAN_JOBS,
            },
            _f,
            indent=2,
        )
    _log(f"\n   [OK] SHAP heatmaps saved -> {shap_json_path}")

    # NPZ with raw per-allele heatmaps
    npz_data = {}
    for allele, hm_dict in shap_results_by_allele.items():
        akey = allele.replace("*", "_").replace(":", "_")
        for L, hm in hm_dict.items():
            npz_data[f"{akey}_{L}"] = np.array(hm)
    if npz_data:
        npz_path = shap_out_dir / "shap_heatmaps.npz"
        np.savez_compressed(npz_path, **npz_data)
        _log(f"   [OK] SHAP NPZ saved      -> {npz_path}")

    if shap_importance_by_allele:
        _log("\n   Position importance (merged, normalised):")
        header = "Allele".ljust(20) + "".join(
            f"  P{i+1:>2}" for i in range(MAX_POSITIONS)
        )
        _log("   " + header)
        _log("   " + "-" * len(header))
        for allele, imp in shap_importance_by_allele.items():
            row = allele.ljust(20) + "".join(f"  {v:.2f}" for v in imp)
            _log("   " + row)

except ImportError as _e:
    _log(f"\n   [WARN] SHAP not available ({_e}); skipping SHAP step.")
except Exception as _e:
    _log(f"\n   [WARN] Direct NetMHCpan SHAP step failed: {_e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# ============================================================================
# STEP 7: Calculate Performance Metrics (Affinity + Mass-spec)
# ============================================================================
_log("\nCalculating performance metrics (standard, leakage-safe)...")

BINDING_THRESHOLD = args.threshold
CLIP = 50000  # nM clipping for log-scale regression

# Ensure numeric
results['measurement_value'] = pd.to_numeric(results['measurement_value'], errors='coerce')
results['predicted_affinity'] = pd.to_numeric(results['predicted_affinity'], errors='coerce')

# Keep only rows with valid predicted affinity (needed for both affinity + MS positives)
results_valid_pred = results[
    np.isfinite(results['predicted_affinity']) & (results['predicted_affinity'] > 0)
].copy()

if len(results_valid_pred) == 0:
    _log("\n[ERROR] No valid predictions to evaluate!")
    sys.exit(1)

# ---------------------------
# A) AFFINITY REGRESSION (IC50) - evaluate only exact quantitative affinity
# ---------------------------
aff_reg = results_valid_pred[
    (results_valid_pred['measurement_kind'] == 'affinity') &
    (results_valid_pred['measurement_type'] == 'quantitative') &
    (results_valid_pred['measurement_inequality'] == '=') &
    np.isfinite(results_valid_pred['measurement_value']) & (results_valid_pred['measurement_value'] > 0)
].copy()

reg_metrics = None
if len(aff_reg) >= 30:
    y = np.clip(aff_reg['measurement_value'].astype(float).values, 1, CLIP)
    p = np.clip(aff_reg['predicted_affinity'].astype(float).values, 1, CLIP)
    ly, lp = np.log10(y), np.log10(p)

    # raw-scale metrics (nM) are harsh but useful for completeness
    r2_raw = r2_score(y, p)
    rmse_nM = np.sqrt(mean_squared_error(y, p))
    mae_nM = mean_absolute_error(y, p)

    reg_metrics = {
        'n_reg': int(len(aff_reg)),
        'r2_raw': float(r2_raw),
        'rmse_nM': float(rmse_nM),
        'mae_nM': float(mae_nM),
        'r2_log10': float(r2_score(ly, lp)),
        'rmse_log10': float(np.sqrt(mean_squared_error(ly, lp))),
        'mae_log10': float(mean_absolute_error(ly, lp)),
        'pearson_log10': float(pearsonr(ly, lp)[0]),
        'spearman_log10': float(spearmanr(ly, lp)[0]),
    }

# ---------------------------
# B) AFFINITY CLASSIFICATION (@threshold nM) - handle censoring (<, >)
# ---------------------------
aff_all = results_valid_pred[
    (results_valid_pred['measurement_kind'] == 'affinity') &
    np.isfinite(results_valid_pred['measurement_value']) & (results_valid_pred['measurement_value'] > 0)
].copy()

def true_label(sample, thr=BINDING_THRESHOLD):
    val = float(sample['measurement_value'])
    ineq = sample.get('measurement_inequality', '=')
    if ineq == '=':
        return 1 if val <= thr else 0
    if ineq == '<':
        # definitely binder only if the upper-bound is already <= threshold
        return 1 if val <= thr else None
    if ineq == '>':
        # definitely non-binder only if the lower-bound is already >= threshold
        return 0 if val >= thr else None
    return None

aff_all['y_true'] = aff_all.apply(true_label, axis=1)
aff_all['y_pred'] = (aff_all['predicted_affinity'].astype(float) <= BINDING_THRESHOLD).astype(int)

valid_cls = aff_all.dropna(subset=['y_true']).copy()
cls_metrics = None
if len(valid_cls) >= 30:
    y_true = valid_cls['y_true'].astype(int).values
    y_pred = valid_cls['y_pred'].astype(int).values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    cls_metrics = {
        'n_cls': int(len(valid_cls)),
        'threshold_nM': float(BINDING_THRESHOLD),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }

# ---------------------------
# C) TABLE I (report-style) - affinity metrics
# ---------------------------
_log("\n" + "="*80)
_log("TABLE I - NETMHCPAN 4.2 PREDICTOR PERFORMANCE (AFFINITY)")
_log("="*80)

rows = []
if reg_metrics:
    rows += [
        ("R^2 (log10 nM, higher better)", f"{reg_metrics['r2_log10']:.3f}"),
        ("Pearson r (log10, higher better)", f"{reg_metrics['pearson_log10']:.3f}"),
        ("Spearman rho (log10, higher better)", f"{reg_metrics['spearman_log10']:.3f}"),
        ("RMSE (log10, lower better)", f"{reg_metrics['rmse_log10']:.3f}"),
        ("MAE (log10, lower better)", f"{reg_metrics['mae_log10']:.3f}"),
        ("N (exact quantitative affinity)", f"{reg_metrics['n_reg']}"),
    ]
else:
    rows += [("Regression note", "Not enough exact quantitative affinity samples")]

if cls_metrics:
    thr = cls_metrics['threshold_nM']
    rows += [
        (f"Accuracy @ {thr:.0f} nM (higher better)", f"{cls_metrics['accuracy']:.3f}"),
        (f"Precision @ {thr:.0f} nM (higher better)", f"{cls_metrics['precision']:.3f}"),
        (f"Recall @ {thr:.0f} nM (higher better)", f"{cls_metrics['recall']:.3f}"),
        (f"F1 @ {thr:.0f} nM (higher better)", f"{cls_metrics['f1']:.3f}"),
        ("N (definite labels after <,>)", f"{cls_metrics['n_cls']}"),
        ("Confusion (TN,FP,FN,TP)", f"{cls_metrics['tn']},{cls_metrics['fp']},{cls_metrics['fn']},{cls_metrics['tp']}"),
    ]
else:
    rows += [("Classification note", "Not enough definite labels after censor handling")]

w = max(len(k) for k, _ in rows)
for k, v in rows:
    _log(f"{k:<{w}}  {v}")
_log("="*80)

# ============================================================================
# STEP 8: Mass-spec ligand evaluation (EL) with decoys
# ============================================================================
_log("\nMass-spec evaluation (EL ligands) with decoys...")

MS_DECOYS_PER_POS = 20          # 20-50 is common; increase for more stable AUC
MS_MAX_DECOYS_PER_ALLELE = 5000 # runtime cap per allele
MS_MIN_POS_PER_ALLELE = 10      # skip tiny groups
RECALL_KS = [10, 50, 100]

rng = np.random.default_rng(42)

ms_pos = results_valid_pred[results_valid_pred['measurement_kind'] == 'mass_spec'].copy()
ms_metrics = None

if len(ms_pos) == 0:
    _log("   (No mass_spec rows found; skipping MS evaluation.)")
else:
    # Build peptide pools by length from the already-filtered evaluation data
    pool_by_len = {L: data[data['peptide'].str.len() == L]['peptide'].unique() for L in range(8, 12)}

    def predict_for_allele(allele_id, pep_seqs):
        """Run NetMHCpan BA mode for (allele, peptides) and return dict peptide->affinity(nM)."""
        if not pep_seqs:
            return {}
        nm_allele = convert_allele_name(allele_id)

        with _tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_fh:
            for p_ in pep_seqs:
                tmp_fh.write(f"{p_}\n")
            tmp_path = tmp_fh.name

        run_env = os.environ.copy()
        run_env['NMHOME'] = str(netmhc_home)
        run_env['NETMHCpan'] = str(netmhc_home)
        run_env['TMPDIR'] = _tempfile.gettempdir()

        out = {}
        try:
            proc = subprocess.run(
                [str(netmhc_bin), '-p', tmp_path, '-a', nm_allele, '-BA'],
                capture_output=True,
                text=True,
                timeout=120,
                env=run_env,
                cwd=str(netmhc_home),
                check=False,
            )

            out_lines = proc.stdout.split('\n')
            hdr_idx = None
            for li, ln in enumerate(out_lines):
                if 'Pos' in ln and 'Peptide' in ln and 'Aff(nM)' in ln:
                    hdr_idx = li
                    break

            if hdr_idx is None:
                return {}

            for ln in out_lines[hdr_idx+2:]:
                if not ln.strip() or ln.strip().startswith('#'):
                    continue
                cols = ln.strip().split()
                if len(cols) >= 16 and cols[0].isdigit():
                    try:
                        out[cols[2]] = float(cols[15])
                    except (ValueError, IndexError):
                        pass

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return out

    aucs, aps = [], []
    recall_at_k = {k: [] for k in RECALL_KS}
    total_pos = 0
    total_decoys = 0

    for allele, grp in ms_pos.groupby('allele'):
        pos_peps = grp['peptide'].drop_duplicates().tolist()
        n_pos = len(pos_peps)
        if n_pos < MS_MIN_POS_PER_ALLELE:
            continue

        # length-matched decoys
        lens = pd.Series(pos_peps).str.len()
        pos_set = set(pos_peps)
        decoys = []

        for L, n_pos_L in lens.value_counts().items():
            L = int(L)
            n_dec_L = min(int(n_pos_L * MS_DECOYS_PER_POS), MS_MAX_DECOYS_PER_ALLELE)
            candidates = pool_by_len.get(L, np.array([], dtype=object))

            cand = np.array([p for p in candidates if p not in pos_set], dtype=object)
            if len(cand) == 0:
                # fallback: shuffled positives (cheap decoys)
                for p in pd.Series(pos_peps)[lens == L]:
                    for _ in range(min(MS_DECOYS_PER_POS, 5)):
                        decoys.append(''.join(rng.permutation(list(p))))
                continue

            replace = len(cand) < n_dec_L
            sampled = rng.choice(cand, size=n_dec_L, replace=replace)
            decoys.extend(sampled.tolist())

        # unique decoys
        seen = set()
        decoys = [d for d in decoys if not (d in seen or seen.add(d))]

        if len(decoys) == 0:
            continue

        if len(decoys) > MS_MAX_DECOYS_PER_ALLELE:
            decoys = rng.choice(decoys, size=MS_MAX_DECOYS_PER_ALLELE, replace=False).tolist()

        # positives already have predictions in results_valid_pred
        pos_aff = results_valid_pred[(results_valid_pred['allele'] == allele) &
                                     (results_valid_pred['measurement_kind'] == 'mass_spec')][['peptide', 'predicted_affinity']]
        pos_aff = pos_aff.drop_duplicates('peptide').set_index('peptide')['predicted_affinity'].to_dict()

        # predict decoys
        dec_aff = predict_for_allele(allele, decoys)

        pos_scored = [(p, float(pos_aff[p])) for p in pos_peps if p in pos_aff and np.isfinite(pos_aff[p]) and pos_aff[p] > 0]
        dec_scored = [(p, float(dec_aff[p])) for p in decoys if p in dec_aff and np.isfinite(dec_aff[p]) and dec_aff[p] > 0]

        if len(pos_scored) < MS_MIN_POS_PER_ALLELE or len(dec_scored) < 50:
            continue

        y_true = np.array([1]*len(pos_scored) + [0]*len(dec_scored))
        # lower affinity = better => score should be higher when affinity is lower
        y_score = np.array([-a for _, a in pos_scored] + [-a for _, a in dec_scored])

        try:
            aucs.append(roc_auc_score(y_true, y_score))
            aps.append(average_precision_score(y_true, y_score))
        except (ValueError, RuntimeError):
            continue

        combined = pos_scored + dec_scored
        combined_sorted = sorted(combined, key=lambda x: x[1])
        ranked = [p for p, _ in combined_sorted]
        pos_set_scored = set([p for p, _ in pos_scored])

        for k in RECALL_KS:
            topk = ranked[:min(k, len(ranked))]
            recall_at_k[k].append(sum(1 for p in topk if p in pos_set_scored) / len(pos_set_scored))

        total_pos += len(pos_scored)
        total_decoys += len(dec_scored)

    if len(aucs) > 0:
        ms_metrics = {
            'n_ms_alleles': int(len(aucs)),
            'n_ms_pos_total': int(total_pos),
            'n_ms_decoys_total': int(total_decoys),
            'ms_roc_auc_macro': float(np.mean(aucs)),
            'ms_pr_auc_macro': float(np.mean(aps)),
        }
        for k in RECALL_KS:
            ms_metrics[f'ms_recall_at_{k}_macro'] = float(np.mean(recall_at_k[k])) if len(recall_at_k[k]) else float('nan')

        _log("\n" + "-"*80)
        _log("TABLE II - MASS-SPEC (EL LIGAND) EVALUATION WITH DECOYS")
        _log("-"*80)
        _log(f"N alleles evaluated: {ms_metrics['n_ms_alleles']}")
        _log(f"Total positives:     {ms_metrics['n_ms_pos_total']}")
        _log(f"Total decoys:        {ms_metrics['n_ms_decoys_total']}")
        _log(f"ROC-AUC (higher better): {ms_metrics['ms_roc_auc_macro']:.3f}")
        _log(f"PR-AUC (higher better):  {ms_metrics['ms_pr_auc_macro']:.3f}")
        for k in RECALL_KS:
            _log(
                f"Recall@{k} (higher better): "
                f"{ms_metrics[f'ms_recall_at_{k}_macro']:.3f}"
            )
        _log("-"*80)
    else:
        _log("   (Not enough MS alleles with valid decoy predictions; skipping MS metrics.)")

# ---------------------------------------------------------------------------
# Legacy metric aliases (for summary CSV + final printouts below)
# ---------------------------------------------------------------------------
r2 = reg_metrics.get('r2_raw', float('nan')) if reg_metrics else float('nan')
r2_log = reg_metrics.get('r2_log10', float('nan')) if reg_metrics else float('nan')
rmse = reg_metrics.get('rmse_nM', float('nan')) if reg_metrics else float('nan')
mae = reg_metrics.get('mae_nM', float('nan')) if reg_metrics else float('nan')
pearson_r = reg_metrics.get('pearson_log10', float('nan')) if reg_metrics else float('nan')
spearman_r = reg_metrics.get('spearman_log10', float('nan')) if reg_metrics else float('nan')
accuracy = cls_metrics.get('accuracy', float('nan')) if cls_metrics else float('nan')
precision = cls_metrics.get('precision', float('nan')) if cls_metrics else float('nan')
recall = cls_metrics.get('recall', float('nan')) if cls_metrics else float('nan')
f1 = cls_metrics.get('f1', float('nan')) if cls_metrics else float('nan')

# STEP 9: Save Results
# ============================================================================
_log("\nSaving results...")

# Ensure output directory exists
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Save detailed predictions
predictions_path = output_dir / 'mhc1_netmhcpan_predictions.csv'
results.to_csv(predictions_path, index=False)
_log(f"   [OK] Detailed predictions: {predictions_path}")

# Save summary
summary = pd.DataFrame([{
    'dataset': data_path.stem,
    'predictor': 'NetMHCpan 4.2',
    'n_samples_total': len(data),
    'n_samples_predicted': len(pred_df),
    'n_samples_valid': len(results),
    'binding_threshold_nM': BINDING_THRESHOLD,
    'r2': r2,
    'r2_log': r2_log,
    'rmse': rmse,
    'mae': mae,
    'pearson_r': pearson_r,
    'spearman_r': spearman_r,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'runtime_minutes': total_time / 60
}])
# Add mass-spec metrics (if computed)
if ms_metrics is not None:
    for k, v in ms_metrics.items():
        summary.loc[0, k] = v

summary_path = output_dir / 'mhc1_netmhcpan_summary.csv'
summary.to_csv(summary_path, index=False)
_log(f"   [OK] Summary: {summary_path}")

# ============================================================================
# STEP 10: Final Summary
# ============================================================================
_log("\n" + "="*80)
_log("[OK] EVALUATION COMPLETE!")
_log("="*80)

_log("\nKey Findings:")
if spearman_r >= 0.85 and accuracy >= 0.90:
    _log("   EXCELLENT performance on combined dataset!")
elif spearman_r >= 0.75 and accuracy >= 0.85:
    _log("   GOOD performance on combined dataset!")
elif spearman_r >= 0.65:
    _log("   [WARN] MODERATE performance - comparable to MHCflurry")
else:
    _log("   [WARN] Performance below expectations")

_log("\nComparison with Previous Results:")
_log("   - MHCflurry (Test): Spearman=0.512, Accuracy=0.814")
_log("   - NetMHCpan (Test): Spearman=0.530, Accuracy=0.875")
_log(f"   - NetMHCpan (MHC1): Spearman={spearman_r:.3f}, Accuracy={accuracy:.3f}")

_log("\nNext Steps:")
_log("   1. Visualize predictions vs actual in a notebook")
_log("   2. Analyze performance by HLA allele type")
_log("   3. Compare quantitative vs qualitative measurements")
_log("   4. Run on larger sample if needed (adjust SAMPLE_SIZE)")

_log("\n" + "="*80)
_log(f"Total runtime: {total_time/60:.1f} minutes")
_log("="*80)
