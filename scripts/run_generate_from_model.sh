#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_generate_from_model.sh --run-id RUN_ID --ar-ckpt PATH --candidate-hla HLA [--shap-json PATH] [--netmhcpan PATH]

Runs:
  generator inference from an existing AR checkpoint for one candidate HLA
  with SHAP-guided refinement
EOF
}

RUN_ID=""
AR_CKPT_INPUT=""
CANDIDATE_HLA=""
SHAP_JSON_INPUT="explainer/shap_results.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --ar-ckpt)
      AR_CKPT_INPUT="${2:-}"
      shift 2
      ;;
    --candidate-hla)
      CANDIDATE_HLA="${2:-}"
      shift 2
      ;;
    --shap-json)
      SHAP_JSON_INPUT="${2:-}"
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
[[ -n "${AR_CKPT_INPUT}" ]] || die "--ar-ckpt is required"
[[ -n "${CANDIDATE_HLA}" ]] || die "--candidate-hla is required"
[[ -n "${SHAP_JSON_INPUT}" ]] || die "--shap-json cannot be empty"

require_file "${IEDB_NPZ}"
resolve_netmhcpan

AR_CKPT="$(resolve_input_file "${AR_CKPT_INPUT}")"
require_file "${AR_CKPT}"
SHAP_JSON="$(resolve_input_file "${SHAP_JSON_INPUT}")"
require_file "${SHAP_JSON}"

cd "${REPO_ROOT}"

GENERATE_OUT="outputs/generate_${RUN_ID}"
SHELL_LOG="outputs/run_generate_from_model_${RUN_ID}.log"

setup_shell_log "${SHELL_LOG}"

print_step "NetMHCpan: ${NETMHCPAN}"
print_step "Using AR checkpoint: ${AR_CKPT}"
print_step "Using SHAP JSON: ${SHAP_JSON}"
print_step "Generating peptides for ${CANDIDATE_HLA}"
"${PYTHON_BIN}" generator/generate.py \
  --data "${IEDB_NPZ}" \
  --out_dir "${GENERATE_OUT}" \
  --alleles "${CANDIDATE_HLA}" \
  --ar_ckpt "${AR_CKPT}" \
  --shap_json "${SHAP_JSON}" \
  --refine_mode shap \
  --netmhcpan "${NETMHCPAN}"

print_step "Generation completed"
printf 'AR model: %s\n' "${AR_CKPT}"
