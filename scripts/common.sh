#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="${SCRIPT_DIR}/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

MHC_CSV="${REPO_ROOT}/data_processing/data/mhc_class1.csv"
MHC_NPZ="${REPO_ROOT}/data_processing/data/mhc_class1_ms_balanced.npz"
IEDB_NPZ="${REPO_ROOT}/data_processing/data/iedb.npz"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

print_step() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

setup_shell_log() {
  local log_path="$1"
  mkdir -p "$(dirname "${log_path}")"
  export SHELL_LOG_PATH="${log_path}"

  if [[ "${SHELL_LOG_INITIALIZED:-0}" == "1" ]]; then
    return
  fi
  export SHELL_LOG_INITIALIZED=1

  # Route shell script output to file only (no terminal echo).
  exec >> "${log_path}" 2>&1
  print_step "Shell script log: ${log_path}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "Required file not found: ${path}"
}

resolve_netmhcpan() {
  if [[ -n "${NETMHCPAN:-}" && -d "${NETMHCPAN}" ]]; then
    export NETMHCPAN
    return
  fi

  local downloads_dir="${HOME}/Downloads/netMHCpan-4.2"
  if [[ -d "${downloads_dir}" ]]; then
    export NETMHCPAN="${downloads_dir}"
    return
  fi

  local sibling="${REPO_ROOT}/../netMHCpan-4.2"
  if [[ -d "${sibling}" ]]; then
    export NETMHCPAN="${sibling}"
    return
  fi

  die "NetMHCpan 4.2 not found. Set NETMHCPAN or place netMHCpan-4.2 next to the repo."
}

resolve_input_file() {
  local raw_path="$1"
  if [[ -f "${raw_path}" ]]; then
    printf '%s\n' "${raw_path}"
    return
  fi

  if [[ -f "${REPO_ROOT}/${raw_path}" ]]; then
    printf '%s\n' "${REPO_ROOT}/${raw_path}"
    return
  fi

  die "File not found: ${raw_path}"
}

find_latest_ar_checkpoint() {
  local base_dir="$1"
  local ar_ckpt

  ar_ckpt="$(find "${base_dir}" -type f -name 'ar_model.pt' -print | sort | tail -n 1 || true)"
  [[ -n "${ar_ckpt}" ]] || die "No ar_model.pt found under ${base_dir}"
  printf '%s\n' "${ar_ckpt}"
}

show_run_log_on_failure() {
  local step_name="$1"
  local log_path="$2"
  local exit_code="$3"

  if [[ -f "${log_path}" ]]; then
    printf '\n%s failed with exit code %s. Showing %s:\n' "${step_name}" "${exit_code}" "${log_path}" >&2
    sed -n '1,240p' "${log_path}" >&2 || true
  else
    printf '\n%s failed with exit code %s. No run log found at %s\n' "${step_name}" "${exit_code}" "${log_path}" >&2
  fi
}
