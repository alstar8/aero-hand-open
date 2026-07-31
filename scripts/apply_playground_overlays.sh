#!/usr/bin/env bash
# Copy Aero Hand Open playground overlays onto the mujoco_playground
# submodule checkout. Safe to re-run (idempotent overwrite).
#
# Usage:
#   ./scripts/apply_playground_overlays.sh
#   # or automatically from setup_mujoco_rl_env.sh / train_mujoco_rl_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAYGROUND_DIR="${REPO_ROOT}/sim_rl/mujoco_playground"
OVERLAY_DIR="${REPO_ROOT}/sim_rl/playground_overlays"

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -d "${OVERLAY_DIR}/mujoco_playground" ]] || die \
  "missing overlays at ${OVERLAY_DIR}"
[[ -f "${PLAYGROUND_DIR}/pyproject.toml" ]] || die \
  "mujoco_playground missing at ${PLAYGROUND_DIR}; run scripts/setup_mujoco_rl_env.sh first"

# Copy tracked overlay payloads (skip README).
while IFS= read -r -d '' src; do
  rel="${src#"${OVERLAY_DIR}/"}"
  [[ "${rel}" == "README.md" ]] && continue
  dst="${PLAYGROUND_DIR}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  cp -f "${src}" "${dst}"
done < <(find "${OVERLAY_DIR}" -type f -print0)

# Quick sanity: sized cube envs must be registered after overlay.
for _env in AeroCubeRotateZAxis38mm AeroCubeRotateZAxis25mm AeroCubeRotateZAxis80mm; do
  if ! grep -q "${_env}" \
    "${PLAYGROUND_DIR}/mujoco_playground/_src/manipulation/__init__.py"; then
    die "overlay apply failed: ${_env} not registered"
  fi
done

log "applied playground overlays from ${OVERLAY_DIR}"
