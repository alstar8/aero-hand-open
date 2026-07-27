#!/usr/bin/env bash
# Infer a trained AeroCubeRotateZAxis PPO policy on a real Aero Hand Open.
#
# Usage:
#   ./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint latest
#   ./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint PATH --dry-run
#   ./scripts/infer_aero_hand_rl.sh --cube-pose zed --checkpoint latest --gpu 1
#
# Uses the mujoco_playground uv env from scripts/setup_mujoco_rl_env.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAYGROUND_DIR="${REPO_ROOT}/sim_rl/mujoco_playground"
VENV_DIR="${PLAYGROUND_DIR}/.venv"
INFER_PY="${SCRIPT_DIR}/infer_aero_hand_rl.py"

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -f "${INFER_PY}" ]] || die "missing ${INFER_PY}"
[[ -d "${VENV_DIR}" ]] || die \
  "uv venv missing at ${VENV_DIR}; run scripts/setup_mujoco_rl_env.sh"
command -v uv >/dev/null 2>&1 || die "uv not found on PATH"

export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
unset LD_LIBRARY_PATH || true

# Let the Python entry find SDK + RL mappings + playground sources.
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/sdk/src:${REPO_ROOT}/ros2/src/aero_hand_open_rl:${PLAYGROUND_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PLAYGROUND_DIR}"

# AeroHand needs pyserial; the playground lockfile does not include it.
if ! uv --no-config run python -c 'import serial' >/dev/null 2>&1; then
  log "installing pyserial into playground uv env (required by aero-open-sdk)"
  uv --no-config pip install 'pyserial>=3.5' --index-url https://pypi.org/simple
fi

log "infer_aero_hand_rl.py $*"
exec uv --no-config run python "${INFER_PY}" "$@"
