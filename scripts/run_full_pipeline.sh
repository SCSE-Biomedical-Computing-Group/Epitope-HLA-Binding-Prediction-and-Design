#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_full_pipeline.sh --run-id RUN_ID [--test-alleles CSV] [--device DEVICE] [--shap-bg-size N] [--shap-fg-size N] [--shap-nsamples N] [--netmhcpan-jobs N] [--netmhcpan PATH]

Runs:
  predictor -> explainer -> generator training -> IEDB AR test

Default IEDB test panel:
  HLA-A*02:01,HLA-A*03:01,HLA-A*24:02,HLA-B*07:02,HLA-B*44:02,HLA-B*57:01,HLA-A*01:01,HLA-B*08:01,HLA-B*27:05,HLA-B*58:01

Speed defaults:
  device=auto, shap-bg-size=64, shap-fg-size=128, shap-nsamples=32, netmhcpan-jobs=4
EOF
}

RUN_ID=""
DEFAULT_TEST_ALLELES="HLA-A*02:01,HLA-A*03:01,HLA-A*24:02,HLA-B*07:02,HLA-B*44:02,HLA-B*57:01,HLA-A*01:01,HLA-B*08:01,HLA-B*27:05,HLA-B*58:01"
TEST_ALLELES="${DEFAULT_TEST_ALLELES}"
SHAP_BG_SIZE="${SHAP_BG_SIZE:-64}"
SHAP_FG_SIZE="${SHAP_FG_SIZE:-128}"
SHAP_NSAMPLES="${SHAP_NSAMPLES:-32}"
NETMHCPAN_JOBS="${NETMHCPAN_JOBS:-4}"
GENERATOR_DEVICE="${GENERATOR_DEVICE:-auto}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --test-alleles)
      TEST_ALLELES="${2:-}"
      shift 2
      ;;
    --shap-bg-size)
      SHAP_BG_SIZE="${2:-}"
      shift 2
      ;;
    --device)
      GENERATOR_DEVICE="${2:-}"
      shift 2
      ;;
    --shap-fg-size)
      SHAP_FG_SIZE="${2:-}"
      shift 2
      ;;
    --shap-nsamples)
      SHAP_NSAMPLES="${2:-}"
      shift 2
      ;;
    --netmhcpan-jobs)
      NETMHCPAN_JOBS="${2:-}"
      shift 2
      ;;
    --netmhcpan)
      export NETMHCPAN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "${RUN_ID}" ]] || die "--run-id is required"
[[ -n "${TEST_ALLELES}" ]] || die "--test-alleles cannot be empty"
[[ "${SHAP_BG_SIZE}" =~ ^[0-9]+$ ]] || die "--shap-bg-size must be a positive integer"
[[ "${SHAP_FG_SIZE}" =~ ^[0-9]+$ ]] || die "--shap-fg-size must be a positive integer"
[[ "${SHAP_NSAMPLES}" =~ ^[0-9]+$ ]] || die "--shap-nsamples must be a positive integer"
[[ "${NETMHCPAN_JOBS}" =~ ^[0-9]+$ ]] || die "--netmhcpan-jobs must be a positive integer"
(( SHAP_BG_SIZE > 0 )) || die "--shap-bg-size must be > 0"
(( SHAP_FG_SIZE > 0 )) || die "--shap-fg-size must be > 0"
(( SHAP_NSAMPLES > 0 )) || die "--shap-nsamples must be > 0"
(( NETMHCPAN_JOBS > 0 )) || die "--netmhcpan-jobs must be > 0"
case "${GENERATOR_DEVICE}" in
  auto|cpu|mps|cuda)
    ;;
  *)
    die "--device must be one of: auto, cpu, mps, cuda"
    ;;
esac

require_file "${MHC_CSV}"
require_file "${MHC_NPZ}"
require_file "${IEDB_NPZ}"
resolve_netmhcpan

export NETMHCPAN_JOBS

cd "${REPO_ROOT}"

PREDICTOR_OUT="outputs/predictor_${RUN_ID}"
EXPLAINER_OUT="outputs/explainer_${RUN_ID}"
GENERATOR_OUT="outputs/generator_${RUN_ID}"
IEDB_TEST_OUT="outputs/iedb_test_${RUN_ID}"
SHELL_LOG="outputs/run_full_pipeline_${RUN_ID}.log"

setup_shell_log "${SHELL_LOG}"

print_step "NetMHCpan: ${NETMHCPAN}"
print_step "NetMHCpan parallel jobs: ${NETMHCPAN_JOBS}"
print_step "Generator device: ${GENERATOR_DEVICE}"
print_step "SHAP budgets: bg=${SHAP_BG_SIZE}, fg=${SHAP_FG_SIZE}, nsamples=${SHAP_NSAMPLES}"
print_step "Running predictor"
if SHAP_BG_SIZE="${SHAP_BG_SIZE}" SHAP_FG_SIZE="${SHAP_FG_SIZE}" SHAP_NSAMPLES="${SHAP_NSAMPLES}" SHAP_MAX_ALLELES=10 NETMHCPAN_JOBS="${NETMHCPAN_JOBS}" "${PYTHON_BIN}" prediction-model/run_netmhcpan_mhc1.py \
  --data "${MHC_CSV}" \
  --output-dir "${PREDICTOR_OUT}" \
  --netmhcpan "${NETMHCPAN}"; then
  :
else
  status=$?
  show_run_log_on_failure "Predictor" "${PREDICTOR_OUT}/run.log" "${status}"
  exit "${status}"
fi

print_step "Running explainer"
if "${PYTHON_BIN}" explainer/shap.py \
  --data "${MHC_NPZ}" \
  --n_background "${SHAP_BG_SIZE}" \
  --n_foreground "${SHAP_FG_SIZE}" \
  --nsamples "${SHAP_NSAMPLES}" \
  --netmhcpan_jobs "${NETMHCPAN_JOBS}" \
  --output_dir "${EXPLAINER_OUT}" \
  --netmhcpan "${NETMHCPAN}"; then
  :
else
  status=$?
  show_run_log_on_failure "Explainer" "${EXPLAINER_OUT}/run.log" "${status}"
  exit "${status}"
fi

SHAP_JSON="${EXPLAINER_OUT}/shap_results.json"
require_file "${SHAP_JSON}"

print_step "Training generator with SHAP guidance"
if "${PYTHON_BIN}" generator/generate.py \
  --data "${MHC_NPZ}" \
  --shap_json "${SHAP_JSON}" \
  --out_dir "${GENERATOR_OUT}" \
  --device "${GENERATOR_DEVICE}" \
  --netmhcpan "${NETMHCPAN}"; then
  :
else
  status=$?
  latest_run_log="$(find "${GENERATOR_OUT}" -type f -name 'run.log' -print | sort | tail -n 1 || true)"
  show_run_log_on_failure "Generator training" "${latest_run_log}" "${status}"
  exit "${status}"
fi

AR_CKPT="$(find_latest_ar_checkpoint "${GENERATOR_OUT}")"
print_step "Trained AR checkpoint: ${AR_CKPT}"

print_step "Running default IEDB AR test"
print_step "IEDB test alleles: ${TEST_ALLELES}"
if "${PYTHON_BIN}" generator/generate.py \
  --data "${IEDB_NPZ}" \
  --shap_json "${SHAP_JSON}" \
  --out_dir "${IEDB_TEST_OUT}" \
  --alleles "${TEST_ALLELES}" \
  --ar_ckpt "${AR_CKPT}" \
  --device "${GENERATOR_DEVICE}" \
  --netmhcpan "${NETMHCPAN}"; then
  :
else
  status=$?
  latest_run_log="$(find "${IEDB_TEST_OUT}" -type f -name 'run.log' -print | sort | tail -n 1 || true)"
  show_run_log_on_failure "Generator IEDB test" "${latest_run_log}" "${status}"
  exit "${status}"
fi

print_step "Full pipeline completed"
printf 'AR model: %s\n' "${AR_CKPT}"
