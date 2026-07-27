#!/usr/bin/env python3
"""Compare AeroCubeRotateZAxis sim rollout vs real infer logs for real2sim gaps.

Dumps sim states/actions from the same checkpoint used by real infer, parses
real action/sim_cmd lines, and reports:
  - action / sim_cmd distribution differences
  - sim sensor proprio vs ctrl (what real approximates)
  - sim<->actuation mapping round-trip error
  - first-N step action distance when starting near home
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
_SDK_SRC = _REPO_ROOT / "sdk" / "src"
_RL_PKG = _REPO_ROOT / "ros2" / "src" / "aero_hand_open_rl"
_PLAYGROUND_DIR = _REPO_ROOT / "sim_rl" / "mujoco_playground"

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

for path in (_SCRIPT_DIR, _SDK_SRC, _RL_PKG, _PLAYGROUND_DIR):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax_pmap_compat  # noqa: E402, F401

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

from aero_hand_open_rl.utils.sim_to_real_mappings import (  # noqa: E402
    actuation_array_to_sim_array,
    sim_array_to_actuation_array,
)
from infer_aero_hand_rl import (  # noqa: E402
    ACTION_SCALE,
    DEFAULT_CTRL,
    ENV_NAME,
    _PROPRIO_REORDER,
    build_obs,
    load_policy,
    resolve_checkpoint,
)

STEP_RE = re.compile(
    r"step=\s*(\d+)\s+cube=(\[[^\]]+\])\s+action=(\[[^\]]+\])\s+sim_cmd=(\[[^\]]+\])"
)
NAMES = ["if", "mf", "rf", "pf", "abd", "th1", "th2"]
OBS_NAMES = ["if", "mf", "rf", "pf", "th1", "th2", "abd"]


def parse_real_log(path: Path) -> dict[str, np.ndarray]:
    steps, actions, cmds = [], [], []
    text = path.read_text(errors="replace")
    for m in STEP_RE.finditer(text):
        steps.append(int(m.group(1)))
        actions.append(ast.literal_eval(m.group(3)))
        cmds.append(ast.literal_eval(m.group(4)))
    if not steps:
        raise SystemExit(f"no real step lines found in {path}")
    return {
        "step": np.asarray(steps, dtype=np.int32),
        "action": np.asarray(actions, dtype=np.float32),
        "sim_cmd": np.asarray(cmds, dtype=np.float32),
    }


def mapping_roundtrip(samples: np.ndarray) -> dict[str, np.ndarray]:
    errs = []
    for row in samples:
        act = sim_array_to_actuation_array(row.tolist())
        back = np.asarray(actuation_array_to_sim_array(act), dtype=np.float32)
        errs.append(back - row)
    err = np.asarray(errs, dtype=np.float32)
    return {
        "mae": np.mean(np.abs(err), axis=0),
        "max": np.max(np.abs(err), axis=0),
        "rmse": np.sqrt(np.mean(err**2, axis=0)),
    }


def run_sim_rollout(
    jit_inference_fn,
    n_steps: int,
    seed: int,
    disable_obs_noise: bool,
) -> dict[str, np.ndarray]:
    from mujoco_playground import registry

    env = registry.load(ENV_NAME)
    if disable_obs_noise:
        env._config.noise_config.level = 0.0

    rng = jax.random.PRNGKey(seed)
    state = jax.jit(env.reset)(rng)
    step_fn = jax.jit(env.step)

    actions = []
    states = []
    ctrls = []
    sensor_proprio = []
    motor_targets = []
    cube_pos = []

    for _ in range(n_steps):
        obs_state = np.asarray(state.obs["state"], dtype=np.float32).ravel()
        states.append(obs_state.copy())
        sensor_proprio.append(obs_state[:7].copy())

        ctrl = np.asarray(state.data.ctrl, dtype=np.float32).ravel()
        ctrls.append(ctrl.copy())
        # Hand NQ=16; cube freejoint qpos = [xyz(3), quat(4)]
        cube_pos.append(np.asarray(state.data.qpos[16:19], dtype=np.float32))

        rng, act_rng = jax.random.split(rng)
        action = np.asarray(jit_inference_fn(state.obs, act_rng)[0], dtype=np.float32).ravel()
        actions.append(action.copy())
        state = step_fn(state, jp.asarray(action))
        motor_targets.append(
            np.asarray(state.info["motor_targets"], dtype=np.float32).ravel()
        )

    return {
        "action": np.asarray(actions, dtype=np.float32),
        "state": np.asarray(states, dtype=np.float32),
        "ctrl": np.asarray(ctrls, dtype=np.float32),
        "sensor_proprio": np.asarray(sensor_proprio, dtype=np.float32),
        "motor_targets": np.asarray(motor_targets, dtype=np.float32),
        "cube_pos": np.asarray(cube_pos, dtype=np.float32),
    }


def summarize_vec(name: str, a: np.ndarray, labels: list[str]) -> None:
    print(f"\n{name}")
    print(f"  shape={a.shape} mean={np.round(a.mean(0), 4).tolist()}")
    print(f"  std ={np.round(a.std(0), 4).tolist()}")
    print(f"  min ={np.round(a.min(0), 4).tolist()}")
    print(f"  max ={np.round(a.max(0), 4).tolist()}")
    print(f"  labels={labels}")


def compare_mats(name: str, sim: np.ndarray, real: np.ndarray, labels: list[str]) -> None:
    n = min(len(sim), len(real))
    s, r = sim[:n], real[:n]
    mae = np.mean(np.abs(s - r), axis=0)
    rmse = np.sqrt(np.mean((s - r) ** 2, axis=0))
    # Also compare distributions regardless of temporal align
    mean_gap = np.abs(sim.mean(0) - real.mean(0))
    std_gap = np.abs(sim.std(0) - real.std(0))
    print(f"\n{name} (aligned first {n} samples)")
    print(f"  MAE /dim ={np.round(mae, 4).tolist()}")
    print(f"  RMSE/dim ={np.round(rmse, 4).tolist()}")
    print(f"  |Δmean|  ={np.round(mean_gap, 4).tolist()}")
    print(f"  |Δstd|   ={np.round(std_gap, 4).tolist()}")
    print(f"  labels   ={labels}")
    print(f"  MAE overall={mae.mean():.4f}  RMSE overall={rmse.mean():.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument(
        "--real-log",
        default=str(
            Path.home()
            / ".cursor/projects/home-admin-aero-hand-open/terminals/2.txt"
        ),
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", default=None)
    parser.add_argument(
        "--disable-obs-noise",
        action="store_true",
        help="Zero env observation noise for cleaner sim dump.",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "sim_rl" / "mujoco_playground" / "logs" / "sim_real_compare.npz"),
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint: {checkpoint}")
    print(f"JAX backend: {jax.default_backend()} devices={jax.devices()}")

    real = parse_real_log(Path(args.real_log))
    print(
        f"Real log: {args.real_log} "
        f"n={len(real['step'])} steps {real['step'][0]}..{real['step'][-1]}"
    )

    print("\n=== Loading policy (once) ===")
    jit_inference_fn = load_policy(checkpoint, seed=args.seed)

    print("\n=== Running sim rollout ===")
    sim = run_sim_rollout(
        jit_inference_fn,
        n_steps=args.steps,
        seed=args.seed,
        disable_obs_noise=args.disable_obs_noise,
    )

    # Ctrl in actuator order; reorder to obs proprio order for comparison.
    ctrl_as_obs = sim["ctrl"][:, _PROPRIO_REORDER]
    sensor = sim["sensor_proprio"]
    ctrl_vs_sensor = sensor - ctrl_as_obs

    print("\n========== DISTRIBUTIONS ==========")
    summarize_vec("SIM action", sim["action"], NAMES)
    summarize_vec("REAL action", real["action"], NAMES)
    summarize_vec("SIM motor_targets (after step)", sim["motor_targets"], NAMES)
    summarize_vec("REAL sim_cmd", real["sim_cmd"], NAMES)
    summarize_vec("SIM sensor proprio (obs order)", sensor, OBS_NAMES)
    summarize_vec("SIM ctrl reordered to obs order", ctrl_as_obs, OBS_NAMES)

    print("\n========== SIM: sensor proprio vs ctrl (proxy real uses) ==========")
    print(f"  MAE /dim ={np.round(np.mean(np.abs(ctrl_vs_sensor), 0), 5).tolist()}")
    print(f"  max /dim ={np.round(np.max(np.abs(ctrl_vs_sensor), 0), 5).tolist()}")
    print(f"  RMSE overall={np.sqrt(np.mean(ctrl_vs_sensor**2)):.5f}")
    print(
        "  NOTE: real build_obs feeds mapped get_actuations() as proprio, "
        "while training uses tendon/joint sensors."
    )

    print("\n========== ACTION / CMD COMPARE ==========")
    # Real log is every 20 steps starting ~60; only use steps inside the sim dump.
    real_steps = real["step"]
    in_range = real_steps < args.steps
    if np.any(in_range):
        rs = real_steps[in_range]
        sim_at_real = sim["action"][rs]
        sim_cmd_at_real = sim["motor_targets"][rs]
        compare_mats(
            "action @ real logged steps",
            sim_at_real,
            real["action"][in_range],
            NAMES,
        )
        compare_mats(
            "sim_cmd @ real logged steps",
            sim_cmd_at_real,
            real["sim_cmd"][in_range],
            NAMES,
        )
    else:
        compare_mats("action (prefix align)", sim["action"], real["action"], NAMES)
        compare_mats(
            "sim_cmd (prefix align)", sim["motor_targets"], real["sim_cmd"], NAMES
        )

    # Distribution-only Wasserstein-ish / overlap via z-score of real mean in sim
    print("\n========== ACTION DISTRIBUTION GAP (not temporally aligned) ==========")
    for i, n in enumerate(NAMES):
        sm, ss = float(sim["action"][:, i].mean()), float(sim["action"][:, i].std() + 1e-8)
        rm, rs = float(real["action"][:, i].mean()), float(real["action"][:, i].std() + 1e-8)
        print(
            f"  {n:4s}: sim μ={sm:+.3f} σ={ss:.3f} | "
            f"real μ={rm:+.3f} σ={rs:.3f} | "
            f"|Δμ|={abs(sm-rm):.3f} |Δσ|={abs(ss-rs):.3f}"
        )

    print("\n========== MAPPING ROUND-TRIP (sim_cmd -> act -> sim) ==========")
    # Sample from home + real cmds + sim motor targets
    samples = np.vstack(
        [
            DEFAULT_CTRL[None, :],
            real["sim_cmd"],
            sim["motor_targets"][:: max(1, len(sim["motor_targets"]) // 50)],
        ]
    ).astype(np.float32)
    rt = mapping_roundtrip(samples)
    print(f"  MAE /dim ={np.round(rt['mae'], 5).tolist()}")
    print(f"  max /dim ={np.round(rt['max'], 5).tolist()}")
    print(f"  labels   ={NAMES}")

    print("\n========== HOME OPEN-LOOP POLICY CHECK ==========")
    # Same as real step 0: proprio≈home ctrl (mapped), last_action=0
    home_obs = build_obs(DEFAULT_CTRL, np.zeros(7, dtype=np.float32))
    rng = jax.random.PRNGKey(args.seed)
    rng, act_rng = jax.random.split(rng)
    home_action = np.asarray(jit_inference_fn(home_obs, act_rng)[0], dtype=np.float32).ravel()
    home_cmd = DEFAULT_CTRL + home_action * ACTION_SCALE
    print(f"  home proprio (ctrl order)={DEFAULT_CTRL.tolist()}")
    print(f"  home obs proprio (sensor order)={DEFAULT_CTRL[_PROPRIO_REORDER].tolist()}")
    print(f"  action@home={np.round(home_action, 4).tolist()}")
    print(f"  sim_cmd@home={np.round(home_cmd, 4).tolist()}")
    print(f"  sim action[0]={np.round(sim['action'][0], 4).tolist()}")
    print(
        f"  |action@home - sim[0]| MAE={np.mean(np.abs(home_action - sim['action'][0])):.4f}"
    )
    # Compare to earliest real action sample
    print(f"  real action@step{int(real['step'][0])}={np.round(real['action'][0], 4).tolist()}")
    print(
        f"  |action@home - real[0]| MAE={np.mean(np.abs(home_action - real['action'][0])):.4f}"
    )

    print("\n========== REAL2SIM RISK FLAGS ==========")
    flags = []
    sens_mae = float(np.mean(np.abs(ctrl_vs_sensor)))
    if sens_mae > 0.01:
        flags.append(
            f"HIGH: sim sensor vs ctrl MAE={sens_mae:.4f} "
            "(real uses mapped actuation≈ctrl, not sensors)"
        )
    else:
        flags.append(f"OK-ish: sim sensor vs ctrl MAE={sens_mae:.4f}")

    rt_mae = float(rt["mae"].mean())
    if rt_mae > 0.005:
        flags.append(f"HIGH: mapping round-trip MAE={rt_mae:.5f}")
    else:
        flags.append(f"OK-ish: mapping round-trip MAE={rt_mae:.5f}")

    act_mean_gap = float(np.mean(np.abs(sim["action"].mean(0) - real["action"].mean(0))))
    if act_mean_gap > 0.2:
        flags.append(f"HIGH: action mean gap overall={act_mean_gap:.3f}")
    else:
        flags.append(f"MOD: action mean gap overall={act_mean_gap:.3f}")

    # Real cube pose is fixed mock; sim cube moves — policy doesn't observe cube,
    # but dynamics differ so proprio/action trajectories diverge.
    cube_travel = float(np.linalg.norm(sim["cube_pos"][-1] - sim["cube_pos"][0]))
    flags.append(
        f"NOTE: sim cube travel={cube_travel:.4f}m over {args.steps} steps; "
        "real used fixed mock cube pose (policy state ignores cube)."
    )
    flags.append(
        "NOTE: real proprio is get_actuations() (command feedback), "
        "not physical tendon length; tracking lag feeds OOD states."
    )
    for f in flags:
        print(f"  - {f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sim_action=sim["action"],
        sim_state=sim["state"],
        sim_ctrl=sim["ctrl"],
        sim_sensor_proprio=sim["sensor_proprio"],
        sim_motor_targets=sim["motor_targets"],
        sim_cube_pos=sim["cube_pos"],
        real_step=real["step"],
        real_action=real["action"],
        real_sim_cmd=real["sim_cmd"],
        home_action=home_action,
        home_cmd=home_cmd,
    )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
