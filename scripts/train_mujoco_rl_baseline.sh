#!/usr/bin/env bash
# Train the Aero Hand Open MuJoCo Playground PPO baseline using the uv env
# created by scripts/setup_mujoco_rl_env.sh.
#
# Usage:
#   ./scripts/train_mujoco_rl_baseline.sh
#   ./scripts/train_mujoco_rl_baseline.sh AeroCubeRotateZAxis38mm
#   ./scripts/train_mujoco_rl_baseline.sh AeroCubeRotateZAxis25mm
#   ./scripts/train_mujoco_rl_baseline.sh --env_name AeroCubeRotateZAxis38mm
#   ./scripts/train_mujoco_rl_baseline.sh --gpu 1
#   ./scripts/train_mujoco_rl_baseline.sh --smoke
#   ./scripts/train_mujoco_rl_baseline.sh --no_tb
#   ./scripts/train_mujoco_rl_baseline.sh --no_domain_randomization
#   ./scripts/train_mujoco_rl_baseline.sh --play_only --load_checkpoint_path PATH
#   ./scripts/train_mujoco_rl_baseline.sh --play_only --load_checkpoint_path latest
#
# Logs / checkpoints land under:
#   sim_rl/mujoco_playground/logs/<env>-<timestamp>[-suffix]/
# TensorBoard is on by default (events under that run dir); use --no_tb to disable.
# Domain randomization is on by default (real-setup-oriented); use
# --no_domain_randomization to disable.

set -euo pipefail

DEFAULT_ENV_NAME="AeroCubeRotateZAxis"

ENV_NAME="${ENV_NAME:-${DEFAULT_ENV_NAME}}"
ENV_NAME_SET=0
# GPU_ID / --gpu pin training to one device via CUDA_VISIBLE_DEVICES.
GPU_ID="${GPU_ID:-}"
SUFFIX=""
PLAY_ONLY=0
LOAD_CHECKPOINT_PATH=""
SMOKE=0
USE_TB=1
USE_WANDB=0
DOMAIN_RANDOMIZATION=1
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Train the MuJoCo Playground PPO baseline for Aero Hand Open.

Uses the existing uv venv at sim_rl/mujoco_playground/.venv (run
scripts/setup_mujoco_rl_env.sh first). Default env is AeroCubeRotateZAxis with
the tuned brax PPO hyperparameters from mujoco_playground.

Proprio defaults to commanded ctrl (matches real get_actuations deploy).
Domain randomization is on by default (friction/mass/actuator spread tuned
toward the real hand).

Task name (optional positional, or --env_name):
  AeroCubeRotateZAxis       50mm cube (default)
  AeroCubeRotateZAxis38mm   38mm cube
  AeroCubeRotateZAxis25mm   25mm cube

Options:
  --env_name NAME              Environment / task name (default: AeroCubeRotateZAxis)
  --gpu ID                     CUDA device id (sets CUDA_VISIBLE_DEVICES)
  --suffix SUFFIX              Experiment name suffix
  --smoke                      Short sanity run (1e6 steps, 256 envs, 2 evals)
  --play_only                  Roll out a checkpoint instead of training
                               (saves rollout*.mp4 under the resolved step dir)
  --load_checkpoint_path PATH  Checkpoint dir/file (required with --play_only).
                               Use "latest" to pick the newest run under
                               logs/<env_name>-*/checkpoints for --env_name.
  --use_tb                     Enable TensorBoard logging (default)
  --no_tb                      Disable TensorBoard logging
  --use_wandb                  Enable Weights & Biases logging
  --domain_randomization       Enable domain randomization (default)
  --no_domain_randomization    Disable domain randomization
  -h, --help                   Show this help

Examples:
  ./scripts/train_mujoco_rl_baseline.sh AeroCubeRotateZAxis38mm --gpu 0
  ./scripts/train_mujoco_rl_baseline.sh AeroCubeRotateZAxis25mm --gpu 0
  ./scripts/train_mujoco_rl_baseline.sh --env_name AeroCubeRotateZAxis38mm --smoke

GPU_ID=0 may be used instead of --gpu. Any args after -- are forwarded to
learning/train_jax_ppo.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env_name)
      ENV_NAME="${2:?--env_name requires a value}"
      ENV_NAME_SET=1
      shift 2
      ;;
    --gpu)
      GPU_ID="${2:?--gpu requires a device id, e.g. 0}"
      shift 2
      ;;
    --suffix)
      SUFFIX="${2:?--suffix requires a value}"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --play_only)
      PLAY_ONLY=1
      shift
      ;;
    --load_checkpoint_path)
      LOAD_CHECKPOINT_PATH="${2:?--load_checkpoint_path requires a path}"
      shift 2
      ;;
    --use_tb)
      USE_TB=1
      shift
      ;;
    --no_tb)
      USE_TB=0
      shift
      ;;
    --use_wandb)
      USE_WANDB=1
      shift
      ;;
    --domain_randomization)
      DOMAIN_RANDOMIZATION=1
      shift
      ;;
    --no_domain_randomization)
      DOMAIN_RANDOMIZATION=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -*)
      # Forward unknown flags to train_jax_ppo.py (e.g. --seed 2).
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      # Positional task / env name (e.g. AeroCubeRotateZAxis38mm).
      if [[ "${ENV_NAME_SET}" -eq 0 ]]; then
        ENV_NAME="$1"
        ENV_NAME_SET=1
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAYGROUND_DIR="${REPO_ROOT}/sim_rl/mujoco_playground"
VENV_DIR="${PLAYGROUND_DIR}/.venv"
TRAIN_SCRIPT="${PLAYGROUND_DIR}/learning/train_jax_ppo.py"

LOGS_DIR="${PLAYGROUND_DIR}/logs"

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

require_env() {
  [[ -f "${PLAYGROUND_DIR}/pyproject.toml" ]] || die \
    "mujoco_playground missing at ${PLAYGROUND_DIR}; run scripts/setup_mujoco_rl_env.sh"
  [[ -d "${VENV_DIR}" ]] || die \
    "uv venv missing at ${VENV_DIR}; run scripts/setup_mujoco_rl_env.sh"
  [[ -f "${TRAIN_SCRIPT}" ]] || die "training script not found: ${TRAIN_SCRIPT}"
  command -v uv >/dev/null 2>&1 || die "uv not found on PATH"
  # Ensure 38mm env + train helpers exist even if submodule is a stock
  # deepmind checkout (overlays are part of this repo, not the submodule).
  "${SCRIPT_DIR}/apply_playground_overlays.sh"
}

# Resolve --load_checkpoint_path latest to the newest checkpointed run for ENV_NAME.
# Prefers run dirs with numeric brax checkpoint steps under checkpoints/.
resolve_latest_checkpoint() {
  local logs_dir="${LOGS_DIR}"
  local prefix="${ENV_NAME}-"
  local run_dir=""
  local ckpt_dir=""
  local candidates=()
  local d

  [[ -d "${logs_dir}" ]] || die \
    "no logs directory at ${logs_dir}; train first or pass an explicit --load_checkpoint_path"

  shopt -s nullglob
  for d in "${logs_dir}/${prefix}"*; do
    [[ -d "${d}" ]] || continue
    candidates+=("${d}")
  done
  shopt -u nullglob

  ((${#candidates[@]})) || die \
    "no runs matching ${logs_dir}/${prefix}* for --load_checkpoint_path latest"

  # Newest first by mtime, then by name (timestamped run dirs sort lexicographically).
  mapfile -t candidates < <(
    printf '%s\n' "${candidates[@]}" | xargs -d '\n' ls -1dt
  )

  for d in "${candidates[@]}"; do
    ckpt_dir="${d}/checkpoints"
    [[ -d "${ckpt_dir}" ]] || continue
    # Require at least one numeric step dir (brax orbax checkpoints).
    if compgen -G "${ckpt_dir}/[0-9]*" >/dev/null; then
      run_dir="${d}"
      break
    fi
  done

  [[ -n "${run_dir}" ]] || die \
    "no checkpoints found under ${logs_dir}/${prefix}*/checkpoints for --load_checkpoint_path latest"

  LOAD_CHECKPOINT_PATH="${run_dir}/checkpoints"
  log "resolved --load_checkpoint_path latest -> ${LOAD_CHECKPOINT_PATH}"
}

# Absolutize checkpoint paths so repo-relative args still work after cd into
# PLAYGROUND_DIR (avoids .../mujoco_playground/sim_rl/mujoco_playground/logs/...).
resolve_checkpoint_path() {
  local raw="${LOAD_CHECKPOINT_PATH}"
  local cand=""
  local prefix="sim_rl/mujoco_playground/"

  [[ -n "${raw}" ]] || return 0
  [[ "${raw}" == "latest" ]] && return 0

  if [[ "${raw}" = /* ]]; then
    [[ -e "${raw}" ]] || die "checkpoint path does not exist: ${raw}"
    LOAD_CHECKPOINT_PATH="$(cd "${raw}" 2>/dev/null && pwd || readlink -f "${raw}")"
    return 0
  fi

  for cand in \
    "${REPO_ROOT}/${raw}" \
    "${PLAYGROUND_DIR}/${raw}" \
    "${PLAYGROUND_DIR}/${raw#"${prefix}"}"; do
    if [[ -e "${cand}" ]]; then
      LOAD_CHECKPOINT_PATH="$(cd "${cand}" 2>/dev/null && pwd || readlink -f "${cand}")"
      log "resolved --load_checkpoint_path -> ${LOAD_CHECKPOINT_PATH}"
      return 0
    fi
  done

  die "checkpoint path does not exist: ${raw}
 tried: ${REPO_ROOT}/${raw}
        ${PLAYGROUND_DIR}/${raw}
        ${PLAYGROUND_DIR}/${raw#"${prefix}"}"
}

build_train_args() {
  TRAIN_ARGS=(--env_name "${ENV_NAME}")

  if [[ -n "${SUFFIX}" ]]; then
    TRAIN_ARGS+=(--suffix "${SUFFIX}")
  elif [[ "${SMOKE}" -eq 1 ]]; then
    TRAIN_ARGS+=(--suffix smoke)
  else
    TRAIN_ARGS+=(--suffix baseline)
  fi

  if [[ "${PLAY_ONLY}" -eq 1 ]]; then
    [[ -n "${LOAD_CHECKPOINT_PATH}" ]] || die \
      "--play_only requires --load_checkpoint_path (path or 'latest')"
    if [[ "${LOAD_CHECKPOINT_PATH}" == "latest" ]]; then
      resolve_latest_checkpoint
    else
      resolve_checkpoint_path
    fi
    TRAIN_ARGS+=(--play_only --load_checkpoint_path "${LOAD_CHECKPOINT_PATH}")
  fi

  if [[ "${SMOKE}" -eq 1 ]]; then
    [[ "${PLAY_ONLY}" -eq 0 ]] || die "--smoke cannot be combined with --play_only"
    # Keep overrides explicit and small so a CPU/GPU box can validate the pipeline.
    TRAIN_ARGS+=(
      --num_timesteps=1000000
      --num_envs=256
      --num_evals=2
      --num_eval_envs=64
      --batch_size=256
      --num_minibatches=8
      --unroll_length=40
      --num_updates_per_batch=4
    )
  fi

  if [[ "${USE_TB}" -eq 1 ]]; then
    TRAIN_ARGS+=(--use_tb)
  fi
  if [[ "${USE_WANDB}" -eq 1 ]]; then
    TRAIN_ARGS+=(--use_wandb)
  fi
  if [[ "${DOMAIN_RANDOMIZATION}" -eq 1 ]]; then
    TRAIN_ARGS+=(--domain_randomization)
  fi

  if ((${#EXTRA_ARGS[@]})); then
    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
  fi
}

select_gpu() {
  if [[ -z "${GPU_ID}" ]]; then
    return
  fi
  [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer, got: ${GPU_ID}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  # JAX sees only the visible device, so it becomes local gpu:0.
  export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
  log "using GPU id ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
}

main() {
  require_env
  build_train_args
  select_gpu

  # Ampere+ GPUs: avoid TF32 matmul precision hurting RL training stability.
  export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
  export PYTHONUNBUFFERED=1
  # Prefer driver CUDA libs over conda when both are present.
  unset LD_LIBRARY_PATH || true
  # Avoid JAX grabbing most of VRAM on shared GPUs (train_jax_ppo also sets this).
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  # Persist XLA compiles across restarts (train_jax_ppo enables the JAX cache).
  # Without this, every cold start sits at ~0% GPU util while ptxas runs on CPU.
  export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${HOME}/.cache/jax}"
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
  # Headless EGL for post-train rollout videos (no X11 / monitor window).
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export MPLBACKEND="${MPLBACKEND:-Agg}"

  cd "${PLAYGROUND_DIR}"

  if [[ "${PLAY_ONLY}" -eq 1 ]]; then
    log "playing ${ENV_NAME} from ${LOAD_CHECKPOINT_PATH}"
  elif [[ "${SMOKE}" -eq 1 ]]; then
    log "smoke-training ${ENV_NAME} (1e6 steps, 256 envs)"
  else
    log "baseline training ${ENV_NAME} (tuned playground PPO config)"
  fi

  log "jax backend check"
  local backend
  backend="$(uv --no-config run python -c 'import jax; print(jax.default_backend())')"
  log "jax backend: ${backend}"
  uv --no-config run python -c 'import jax; print("devices:", jax.devices())' || true
  if [[ -n "${GPU_ID}" && "${backend}" != "gpu" ]]; then
    die "requested --gpu ${GPU_ID} but JAX backend is '${backend}' (expected gpu). Re-run scripts/setup_mujoco_rl_env.sh so jax[cuda12] is installed after uv sync."
  fi

  log "running: python learning/train_jax_ppo.py ${TRAIN_ARGS[*]}"
  # Bootstrap applies jax.device_put_replicated compat for brax on JAX>=0.11.
  exec uv --no-config run python "${SCRIPT_DIR}/run_train_jax_ppo.py" "${TRAIN_ARGS[@]}"
}

main
