#!/usr/bin/env bash
# Set up a uv virtualenv for MuJoCo Playground RL training (Aero Hand Open).
#
# Usage:
#   ./scripts/setup_mujoco_rl_env.sh           # CUDA 12 JAX (recommended)
#   ./scripts/setup_mujoco_rl_env.sh --cpu     # CPU-only JAX
#
# After setup, activate and train:
#   source sim_rl/mujoco_playground/.venv/bin/activate
#   cd sim_rl/mujoco_playground
#   train-jax-ppo --env_name AeroCubeRotateZAxis
#   # or: uv --no-config run train-jax-ppo --env_name AeroCubeRotateZAxis

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
USE_CUDA=1

usage() {
  cat <<'EOF'
Set up a uv environment for MuJoCo Playground RL training.

Options:
  --cpu              Install CPU-only JAX (skip jax[cuda12])
  --python VERSION   Python version for the venv (default: 3.12)
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      USE_CUDA=0
      shift
      ;;
    --python)
      PYTHON_VERSION="${2:?--python requires a version, e.g. 3.12}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAYGROUND_DIR="${REPO_ROOT}/sim_rl/mujoco_playground"

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "uv found: $(uv --version)"
    return
  fi
  log "uv not found; installing via astral.sh"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  if [[ -f "${HOME}/.local/bin/env" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env"
  elif [[ -f "${HOME}/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.cargo/env"
  fi
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; open a new shell and re-run"
  log "uv installed: $(uv --version)"
}

is_git_checkout() {
  # Submodule checkouts use a .git file; normal clones use a .git directory.
  [[ -e "${1}/.git" ]]
}

# True when the path looks like a submodule checkout but has no usable tree
# (common after a failed/interrupted `git submodule update`).
is_broken_submodule_checkout() {
  [[ -d "${1}" ]] && is_git_checkout "${1}" && [[ ! -f "${1}/pyproject.toml" ]]
}

clear_playground_path() {
  local backup_dir
  backup_dir="${PLAYGROUND_DIR}.pre-submodule-backup.$(date +%Y%m%d-%H%M%S)"
  log "clearing ${PLAYGROUND_DIR} (backup: ${backup_dir})"
  mkdir -p "${backup_dir}"
  shopt -s dotglob nullglob
  local entries=("${PLAYGROUND_DIR}"/*)
  if ((${#entries[@]})); then
    mv "${entries[@]}" "${backup_dir}/"
  fi
  shopt -u dotglob nullglob
  rmdir "${PLAYGROUND_DIR}" 2>/dev/null || true
  if [[ -d "${PLAYGROUND_DIR}" ]]; then
    die "could not clear ${PLAYGROUND_DIR} for submodule clone; remove it manually and re-run"
  fi
}

# Fetch only the commit recorded by the parent repo (~tens of MB vs full history).
# Returns 0 on success, 1 on recoverable failure (caller may fall back).
shallow_init_playground_submodule() {
  local pinned url module_gitdir
  pinned="$(git -C "${REPO_ROOT}" ls-tree HEAD sim_rl/mujoco_playground | awk '{print $3}')"
  [[ -n "${pinned}" ]] || return 1
  url="$(git -C "${REPO_ROOT}" config --file .gitmodules --get submodule.mujoco_playground.url)"
  [[ -n "${url}" ]] || return 1
  module_gitdir="${REPO_ROOT}/.git/modules/mujoco_playground"

  log "shallow-fetching mujoco_playground @ ${pinned}"
  rm -rf "${PLAYGROUND_DIR}" "${module_gitdir}"
  mkdir -p "${PLAYGROUND_DIR}"
  git -C "${PLAYGROUND_DIR}" init -q || return 1
  git -C "${PLAYGROUND_DIR}" remote add origin "${url}" || return 1
  git -C "${PLAYGROUND_DIR}" fetch --depth 1 origin "${pinned}" || return 1
  git -C "${PLAYGROUND_DIR}" checkout --force -q FETCH_HEAD || return 1

  # Convert to the standard submodule gitdir layout.
  mkdir -p "${REPO_ROOT}/.git/modules"
  rm -rf "${module_gitdir}"
  mv "${PLAYGROUND_DIR}/.git" "${module_gitdir}" || return 1
  printf 'gitdir: ../../.git/modules/mujoco_playground\n' >"${PLAYGROUND_DIR}/.git"
  git -C "${module_gitdir}" config core.worktree ../../../sim_rl/mujoco_playground || return 1
  [[ -f "${PLAYGROUND_DIR}/pyproject.toml" ]] || return 1
}

ensure_submodule() {
  if [[ -f "${PLAYGROUND_DIR}/pyproject.toml" ]]; then
    return
  fi
  log "mujoco_playground submodule missing or empty; initializing"
  if [[ ! -d "${REPO_ROOT}/.git" ]]; then
    die "expected git repo at ${REPO_ROOT} and submodule at sim_rl/mujoco_playground"
  fi

  # Broken empty checkouts (gitdir present, no tree) cause:
  #   fatal: Unable to find current revision in submodule path '...'
  # Wipe them (and any non-git leftover content) before re-init.
  if is_broken_submodule_checkout "${PLAYGROUND_DIR}"; then
    log "broken empty submodule checkout detected; resetting"
    git -C "${REPO_ROOT}" submodule deinit -f sim_rl/mujoco_playground 2>/dev/null || true
    rm -rf "${REPO_ROOT}/.git/modules/mujoco_playground"
    clear_playground_path
  elif [[ -d "${PLAYGROUND_DIR}" ]]; then
    # git submodule clone fails if the path exists and is non-empty (e.g. leftover
    # local logs under learning/). Move obstructing content aside first.
    clear_playground_path
  fi

  git -C "${REPO_ROOT}" submodule sync --recursive
  # Prefer a shallow fetch of the pinned SHA: full clone is ~340MB and often
  # times out / leaves an empty checkout that fails on the next run.
  if ! shallow_init_playground_submodule; then
    log "shallow init failed; falling back to full submodule update"
    git -C "${REPO_ROOT}" submodule update --init --recursive sim_rl/mujoco_playground
  fi
  [[ -f "${PLAYGROUND_DIR}/pyproject.toml" ]] || die "failed to initialize sim_rl/mujoco_playground"
}

setup_venv() {
  cd "${PLAYGROUND_DIR}"
  log "creating venv with Python ${PYTHON_VERSION} in ${PLAYGROUND_DIR}/.venv"
  uv venv --python "${PYTHON_VERSION}"

  # shellcheck disable=SC1091
  source .venv/bin/activate

  # Sync first. The lockfile pins CPU jax/jaxlib; installing jax[cuda12]
  # before sync gets overwritten and leaves training on CPU.
  log "syncing playground dependencies (all extras)"
  uv --no-config sync --all-extras

  if [[ "${USE_CUDA}" -eq 1 ]]; then
    log "installing JAX with CUDA 12 support (after sync so plugins stick)"
    # Prefer driver CUDA libs over conda when both are present.
    unset LD_LIBRARY_PATH || true
    uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
  else
    log "installing CPU-only JAX (--cpu)"
    uv pip install -U jax --index-url https://pypi.org/simple
  fi

  patch_mjx_nconmax_compat

  log "verifying mujoco_playground import"
  uv --no-config run python -c "import mujoco_playground; print('mujoco_playground OK')"

  if [[ "${USE_CUDA}" -eq 1 ]]; then
    log "checking JAX backend (expect 'gpu')"
    unset LD_LIBRARY_PATH || true
    local backend
    backend="$(uv --no-config run python -c "import jax; print(jax.default_backend())")"
    log "jax backend: ${backend}"
    if [[ "${backend}" != "gpu" ]]; then
      die "expected JAX GPU backend after jax[cuda12] install, got '${backend}'. Check NVIDIA driver (nvidia-smi) and re-run without conda LD_LIBRARY_PATH."
    fi
  fi
}

# Pinned playground still passes nconmax= into mjx.make_data; MuJoCo >=3.9
# renamed that kwarg to naconmax. Patch the local checkout if needed.
patch_mjx_nconmax_compat() {
  local target="${PLAYGROUND_DIR}/mujoco_playground/_src/mjx_env.py"
  [[ -f "${target}" ]] || die "missing ${target}"
  if grep -q 'contact_max = naconmax if naconmax is not None else nconmax' "${target}"; then
    log "mjx nconmax/naconmax compatibility patch already applied"
    return
  fi
  log "patching mjx_env.make_data for MuJoCo >=3.9 (nconmax -> naconmax)"
  uv --no-config run python - <<'PY'
from pathlib import Path
import re

path = Path("mujoco_playground/_src/mjx_env.py")
text = path.read_text()
if "import inspect\n" not in text:
    text = text.replace("import abc\n", "import abc\nimport inspect\n", 1)

old = '''def make_data(
    model: mujoco.MjModel,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
    impl: Optional[str] = None,
    nconmax: Optional[int] = None,
    njmax: Optional[int] = None,
    device: Optional[jax.Device] = None,
) -> mjx.Data:
  """Initialize MJX Data."""
  data = mjx.make_data(
      model, impl=impl, nconmax=nconmax, njmax=njmax, device=device
  )'''

new = '''def make_data(
    model: mujoco.MjModel,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
    impl: Optional[str] = None,
    nconmax: Optional[int] = None,
    naconmax: Optional[int] = None,
    naccdmax: Optional[int] = None,
    njmax: Optional[int] = None,
    device: Optional[jax.Device] = None,
) -> mjx.Data:
  """Initialize MJX Data.

  MuJoCo >= 3.9 removed ``nconmax`` from ``mjx.make_data`` in favor of
  ``naconmax``. Accept both names and forward the one supported by the
  installed mjx.
  """
  contact_max = naconmax if naconmax is not None else nconmax
  make_kwargs: Dict[str, Any] = {
      "impl": impl,
      "njmax": njmax,
      "device": device,
  }
  params = inspect.signature(mjx.make_data).parameters
  if "naconmax" in params:
    make_kwargs["naconmax"] = contact_max
  elif "nconmax" in params:
    make_kwargs["nconmax"] = contact_max
  if "naccdmax" in params and naccdmax is not None:
    make_kwargs["naccdmax"] = naccdmax
  data = mjx.make_data(model, **make_kwargs)'''

if old not in text:
    raise SystemExit("mjx_env.make_data pattern not found; update patch_mjx_nconmax_compat")
path.write_text(text.replace(old, new, 1))
print("patched", path)
PY
}

print_next_steps() {
  cat <<EOF

Setup complete.

Activate the environment:
  source ${PLAYGROUND_DIR}/.venv/bin/activate

Train the Aero Hand cube Z-rotation baseline (preferred):
  ${REPO_ROOT}/scripts/train_mujoco_rl_baseline.sh --gpu 0

Or manually:
  cd ${PLAYGROUND_DIR}
  export JAX_DEFAULT_MATMUL_PRECISION=highest
  unset LD_LIBRARY_PATH
  uv --no-config run python learning/train_jax_ppo.py --env_name AeroCubeRotateZAxis --suffix baseline
EOF
}

main() {
  ensure_uv
  ensure_submodule
  setup_venv
  print_next_steps
}

main
