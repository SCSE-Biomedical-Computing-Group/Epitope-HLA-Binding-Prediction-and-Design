#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_no_xai_ablation.sh [--run-id RUN_ID] [--netmhcpan PATH]

Trains the generator without SHAP guidance on mhc_class1_ms_balanced.npz,
then tests the trained AR checkpoint on iedb.npz.
EOF
}

RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
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

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="no_xai_ablation_$(date '+%Y%m%d_%H%M%S')"
fi

require_file "${MHC_NPZ}"
require_file "${IEDB_NPZ}"
resolve_netmhcpan

cd "${REPO_ROOT}"

GENERATOR_OUT="outputs/generator_${RUN_ID}"
IEDB_TEST_OUT="outputs/iedb_test_${RUN_ID}"
SHELL_LOG="outputs/run_no_xai_ablation_${RUN_ID}.log"

setup_shell_log "${SHELL_LOG}"

print_step "NetMHCpan: ${NETMHCPAN}"
print_step "Training generator without SHAP guidance"
if "${PYTHON_BIN}" generator/generate.py \
  --data "${MHC_NPZ}" \
  --out_dir "${GENERATOR_OUT}" \
  --refine_mode without_shap \
  --netmhcpan "${NETMHCPAN}"; then
  :
else
  status=$?
  latest_run_log="$(find "${GENERATOR_OUT}" -type f -name run.log -print | sort | tail -n 1 || true)"
  show_run_log_on_failure "No-XAI generator training" "${latest_run_log}" "${status}"
  exit "${status}"
fi

AR_CKPT="$(find_latest_ar_checkpoint "${GENERATOR_OUT}")"
print_step "Trained AR checkpoint: ${AR_CKPT}"

print_step "Testing generator on IEDB data without SHAP guidance"
if "${PYTHON_BIN}" generator/generate.py \
  --data "${IEDB_NPZ}" \
  --out_dir "${IEDB_TEST_OUT}" \
  --alleles all \
  --ar_ckpt "${AR_CKPT}" \
  --refine_mode without_shap \
  --netmhcpan "${NETMHCPAN}"; then
  :
else
  status=$?
  latest_run_log="$(find "${IEDB_TEST_OUT}" -type f -name run.log -print | sort | tail -n 1 || true)"
  show_run_log_on_failure "No-XAI IEDB test" "${latest_run_log}" "${status}"
  exit "${status}"
fi

print_step "No-XAI ablation run completed"
printf 'AR model: %s\n' "${AR_CKPT}"
