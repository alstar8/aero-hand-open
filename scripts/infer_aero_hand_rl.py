#!/usr/bin/env python3
# Copyright 2025 TetherIA, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a trained AeroCubeRotateZAxis PPO policy on a real Aero Hand Open.

By default runs on-board homing/calibration, verifies full open-palm, moves to
the sim home pose (partial MCP curl by design), then closes the control loop
at ctrl_dt. Proprio defaults to last commanded ctrl — matching training's
``proprio_source="ctrl"``. Optional ``--cmd-bias`` adds sim-to-real offsets
(finger curl + thumb abduction) before mapping to hardware.

Cube pose comes from ``--cube-pose mock`` (hardcoded near the training reset)
or ``--cube-pose zed`` (stub for now). The policy observes proprio + last
action only (14D ``state``); cube pose is logged for placement guidance.

Usage:
  ./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint latest
  ./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint PATH --dry-run
  ./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint PATH --no-calibrate
  ./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint PATH --vel 0.1
  ./scripts/infer_aero_hand_rl.sh --cube-pose zed --checkpoint latest --gpu 1

Requires the mujoco_playground uv env (setup_mujoco_rl_env.sh) and aero-open-sdk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
_SDK_SRC = _REPO_ROOT / "sdk" / "src"
_RL_PKG = _REPO_ROOT / "ros2" / "src" / "aero_hand_open_rl"
_PLAYGROUND_DIR = _REPO_ROOT / "sim_rl" / "mujoco_playground"
_LOGS_DIR = _PLAYGROUND_DIR / "logs"

# Headless / compat before JAX / mujoco imports.
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

# ---------------------------------------------------------------------------
# Sim-aligned defaults (AeroCubeRotateZAxis / scene_mjx_cube.xml home keyframe)
# ---------------------------------------------------------------------------

ENV_NAME = "AeroCubeRotateZAxis"

# XML home ctrl: [index, middle, ring, pinky, thumb_abd, th1, th2]
DEFAULT_CTRL = np.array(
    [0.09, 0.09, 0.09, 0.09, 0.75, 0.035, 0.1], dtype=np.float32
)

# Fallback if checkpoint config.json is missing (rotate_z.default_config).
# Prefer checkpoints/config.json written at train time — half_action_scale
# runs store [0.01, 0.01, 0.01, 0.01, 0.35, 0.0015, 0.006].
ACTION_SCALE = np.array(
    [0.02, 0.02, 0.02, 0.02, 0.7, 0.003, 0.012], dtype=np.float32
)

CTRL_DT = 0.05

# rotate_z.reset() nominal cube start (noise stripped) — close to training.
MOCK_CUBE_POS = np.array([0.1, 0.0, 0.05], dtype=np.float32)
# Scene home keyframe cube quat (wxyz).
MOCK_CUBE_QUAT = np.array(
    [0.810967, -0.00262895, -0.585086, -0.000254303], dtype=np.float32
)

OPEN_PALM_ACTUATION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Added to sim_cmd before mapping to real actuations (and subtracted from
# actuator-derived proprio).
#
# Index slack/coupling is handled by a measured real MCP→motor calibration in
# sim_to_real_mappings. Remaining bias is thumb tip spacing only; middle/ring
# keep a small extra curl for underactuated lag.
# Order: [index, middle, ring, pinky, thumb_abd, th1, th2]
DEFAULT_CMD_BIAS = np.array(
    [0.0, -0.007, -0.007, 0.0, 0.25, 0.0, 0.0], dtype=np.float32
)

# Proprio reorder: sim ctrl order -> sensor order (tendons then thumb abd).
# ctrl: [if, mf, rf, pf, abd, th1, th2] -> obs: [if, mf, rf, pf, th1, th2, abd]
_PROPRIO_REORDER = np.array([0, 1, 2, 3, 5, 6, 4], dtype=np.int32)

ESP32_BY_ID_PREFIX = "usb-Espressif_USB_JTAG_serial_debug_unit_"
SERIAL_BY_ID_DIR = "/dev/serial/by-id"


@dataclass(frozen=True)
class CubePose:
    """Cube pose in the hand / palm frame used by the sim (meters, wxyz)."""

    position: np.ndarray  # (3,)
    quaternion_wxyz: np.ndarray  # (4,)
    source: str


class CubePoseProvider(ABC):
    @abstractmethod
    def get_pose(self) -> CubePose:
        raise NotImplementedError


class MockCubePoseProvider(CubePoseProvider):
    """Hardcoded cube pose near the training reset / scene home."""

    def __init__(
        self,
        position: Optional[np.ndarray] = None,
        quaternion_wxyz: Optional[np.ndarray] = None,
    ) -> None:
        self._pose = CubePose(
            position=np.asarray(
                position if position is not None else MOCK_CUBE_POS,
                dtype=np.float32,
            ),
            quaternion_wxyz=np.asarray(
                quaternion_wxyz if quaternion_wxyz is not None else MOCK_CUBE_QUAT,
                dtype=np.float32,
            ),
            source="mock",
        )

    def get_pose(self) -> CubePose:
        return self._pose


class ZedCubePoseProvider(CubePoseProvider):
    """Placeholder for Stereolabs ZED cube tracking.

    Not implemented yet — raises on use so ``--cube-pose zed`` is explicit.
    """

    def __init__(self, serial_number: Optional[str] = None) -> None:
        self.serial_number = serial_number
        try:
            import pyzed.sl as sl  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "ZED cube pose requested but pyzed is not installed.\n"
                "Install the ZED SDK + Python API, then implement "
                "ZedCubePoseProvider.get_pose(), or use --cube-pose mock."
            ) from exc
        raise SystemExit(
            "ZED cube pose provider is stubbed. Implement get_pose() with your "
            "cube detection, or use --cube-pose mock for now."
        )

    def get_pose(self) -> CubePose:
        raise NotImplementedError("ZED cube pose not implemented")


def list_hand_ports() -> list[str]:
    if not os.path.isdir(SERIAL_BY_ID_DIR):
        return []
    return sorted(
        os.path.join(SERIAL_BY_ID_DIR, name)
        for name in os.listdir(SERIAL_BY_ID_DIR)
        if ESP32_BY_ID_PREFIX in name
    )


def resolve_checkpoint(path_or_latest: str, env_name: str = ENV_NAME) -> Path:
    """Resolve ``latest`` or a path to an Orbax checkpoint *step* directory.

    Relative paths are tried against cwd, the repo root, and the playground
    dir (the infer wrapper ``cd``s into playground before launch).
    """
    if path_or_latest != "latest":
        raw = Path(path_or_latest).expanduser()
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend(
                [
                    Path.cwd() / raw,
                    _REPO_ROOT / raw,
                    _PLAYGROUND_DIR / raw,
                ]
            )
            # Repo-relative path used while already inside playground.
            parts = raw.parts
            marker = ("sim_rl", "mujoco_playground")
            if len(parts) >= 2 and parts[:2] == marker:
                candidates.append(_PLAYGROUND_DIR / Path(*parts[2:]))

        ckpt = None
        for cand in candidates:
            resolved = cand.resolve()
            if resolved.exists():
                ckpt = resolved
                break
        if ckpt is None:
            tried = ", ".join(str(c.resolve()) for c in candidates)
            raise SystemExit(
                f"checkpoint path does not exist: {path_or_latest}\n tried: {tried}"
            )
        # Allow pointing at logs/.../checkpoints or a numeric step dir.
        if ckpt.is_dir() and ckpt.name == "checkpoints":
            return _latest_step_dir(ckpt)
        if ckpt.is_dir() and (
            ckpt.name.isdigit() or any(ckpt.glob("*/_METADATA"))
        ):
            return ckpt
        # Parent may be the run dir.
        nested = ckpt / "checkpoints"
        if nested.is_dir():
            return _latest_step_dir(nested)
        return ckpt

    if not _LOGS_DIR.is_dir():
        raise SystemExit(
            f"no logs at {_LOGS_DIR}; train first or pass --checkpoint PATH"
        )

    runs = sorted(
        (p for p in _LOGS_DIR.glob(f"{env_name}-*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        ckpt_root = run / "checkpoints"
        if not ckpt_root.is_dir():
            continue
        try:
            return _latest_step_dir(ckpt_root)
        except SystemExit:
            continue
    raise SystemExit(
        f"no numeric checkpoints under {_LOGS_DIR}/{env_name}-*/checkpoints"
    )


def _latest_step_dir(ckpt_root: Path) -> Path:
    steps = []
    for child in ckpt_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(child)
    if not steps:
        raise SystemExit(f"no numeric step dirs in {ckpt_root}")
    steps.sort(key=lambda p: int(p.name))
    return steps[-1]


def find_checkpoint_config(checkpoint_step_dir: Path) -> Optional[Path]:
    """Locate train-time env config.json next to an Orbax step dir."""
    # Usual layout: logs/<run>/checkpoints/{config.json, 000123/...}
    candidates = [
        checkpoint_step_dir.parent / "config.json",
        checkpoint_step_dir / "config.json",
    ]
    if checkpoint_step_dir.parent.name == "checkpoints":
        candidates.append(checkpoint_step_dir.parent.parent / "config.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_action_scale(
    checkpoint_step_dir: Path,
    env_name: str,
    action_scale_mult: Optional[float] = None,
) -> np.ndarray:
    """Load action_scale from checkpoint config, else env default.

    Training with ``--action_scale_mult 0.5`` writes the scaled vector into
    ``checkpoints/config.json``. Deploy must use that same scale or motor
    targets are 2x too large and the policy fixed-point looks wrong on hardware.
    """
    cfg_path = find_checkpoint_config(checkpoint_step_dir)
    scale: Optional[np.ndarray] = None
    source = "hardcoded fallback"

    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as fp:
            env_cfg = json.load(fp)
        if "action_scale" in env_cfg:
            scale = np.asarray(env_cfg["action_scale"], dtype=np.float32)
            source = str(cfg_path)

    if scale is None:
        try:
            from mujoco_playground import registry

            env_cfg = registry.get_default_config(env_name)
            if "action_scale" in env_cfg:
                scale = np.asarray(env_cfg.action_scale, dtype=np.float32)
                source = f"registry default ({env_name})"
        except Exception as exc:  # noqa: BLE001 — fall back below
            print(f"warn: could not load env action_scale from registry: {exc}")

    if scale is None:
        scale = ACTION_SCALE.copy()

    if scale.shape != (7,):
        raise SystemExit(
            f"action_scale must have 7 values, got shape {scale.shape} from {source}"
        )

    if action_scale_mult is not None:
        scale = scale * float(action_scale_mult)
        source = f"{source} * --action-scale-mult={action_scale_mult}"

    print(f"action_scale ({source})={scale.tolist()}")
    return scale.astype(np.float32)


def build_obs(sim_proprio7: np.ndarray, last_action: np.ndarray) -> dict:
    """Match deploy / rotate_z policy ``state`` (14D)."""
    proprio = np.asarray(sim_proprio7, dtype=np.float32).ravel()
    if proprio.shape != (7,):
        raise ValueError(f"expected 7D sim proprio, got {proprio.shape}")
    reordered = proprio[_PROPRIO_REORDER]
    last_action = np.asarray(last_action, dtype=np.float32).ravel()
    if last_action.shape != (7,):
        raise ValueError(f"expected 7D last_action, got {last_action.shape}")
    return {"state": np.concatenate([reordered, last_action], axis=0)}


def load_policy(
    checkpoint_step_dir: Path,
    seed: int = 1,
    env_name: str = ENV_NAME,
):
    """Load Orbax PPO params via brax train(num_timesteps=0, restore=...)."""
    import functools

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry
    from mujoco_playground import wrapper
    from mujoco_playground.config import manipulation_params

    env = registry.load(env_name)
    ppo_params = manipulation_params.brax_ppo_config(env_name)
    # ConfigDict has no .pop; copy to a plain dict for brax kwargs.
    training_params = dict(ppo_params)
    network_factory_kwargs = dict(training_params.pop("network_factory", {}))
    training_params.pop("num_timesteps", None)
    # Keep restore-only load cheap / deterministic.
    training_params["num_evals"] = 1
    training_params.pop("num_resets_per_eval", None)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_factory_kwargs
    )

    print(f"Loading policy from {checkpoint_step_dir} ...")
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        num_timesteps=0,
        seed=seed,
        restore_checkpoint_path=str(checkpoint_step_dir),
        **training_params,
    )
    inference_fn = make_inference_fn(params, deterministic=True)
    return jax.jit(inference_fn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer an AeroCubeRotateZAxis PPO checkpoint on a real Aero Hand Open."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help='Orbax step dir, run dir, checkpoints/, or "latest" (default).',
    )
    parser.add_argument(
        "--env_name",
        default=ENV_NAME,
        help=(
            "Playground env / task name (default: AeroCubeRotateZAxis). "
            "Sized cubes: AeroCubeRotateZAxis25mm / 38mm / 80mm."
        ),
    )
    parser.add_argument(
        "--cube-pose",
        choices=("mock", "zed"),
        default="mock",
        help="Cube pose source (default: mock near training reset).",
    )
    parser.add_argument(
        "--zed-serial",
        default=None,
        help="Optional ZED camera serial (for --cube-pose zed).",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port (auto-detect Espressif USB-JTAG if omitted).",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List candidate hand serial ports and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No hardware: print commands using mock proprio = last command.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Policy runtime in seconds after the start hold (default: 30).",
    )
    parser.add_argument(
        "--start-hold",
        type=float,
        default=2.0,
        help="Seconds to hold sim home pose before policy (place the cube).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=CTRL_DT,
        help=f"Policy control period seconds (default: {CTRL_DT}).",
    )
    parser.add_argument(
        "--vel",
        type=float,
        default=1.0,
        help=(
            "Wall-clock time scale for the policy loop (default: 1.0). "
            "Use --vel 0.1 to run at 0.1x speed (10x slower), giving fingers "
            "more time to reach each pose. Step count still uses --duration/--dt; "
            "wall period is dt/vel."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA device id (sets CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--open-on-exit",
        action="store_true",
        help="Send open-palm actuations on exit (default: hold last command).",
    )
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run on-board homing/calibration before the policy loop "
            "(default: on). Use --no-calibrate to skip."
        ),
    )
    parser.add_argument(
        "--calibrate-timeout",
        type=float,
        default=175.0,
        help="Seconds to wait for homing ACK (default: 175).",
    )
    parser.add_argument(
        "--verify-open",
        type=float,
        default=1.5,
        help=(
            "After homing, hold full open-palm (0°) for this many seconds "
            "before moving to sim home (default: 1.5). Set 0 to skip. "
            "Sim home intentionally leaves fingers curled (~80° MCP); that is "
            "not a failed pinky home."
        ),
    )
    parser.add_argument(
        "--cmd-bias",
        default=",".join(str(float(x)) for x in DEFAULT_CMD_BIAS),
        help=(
            "Comma-separated 7-float bias added to sim_cmd before mapping "
            f"(default: {','.join(str(float(x)) for x in DEFAULT_CMD_BIAS)}). "
            "Finger tendons: negative shortens cable (more curl). "
            "thumb_abd>+0 brings thumb closer to index on hardware."
        ),
    )
    parser.add_argument(
        "--proprio-from-actuators",
        action="store_true",
        help=(
            "Build proprio from get_actuations() instead of last commanded "
            "ctrl (default: commanded ctrl, matching training "
            "proprio_source=ctrl)."
        ),
    )
    parser.add_argument(
        "--action-scale-mult",
        type=float,
        default=None,
        help=(
            "Optional extra multiplier on the loaded action_scale. Prefer "
            "relying on checkpoints/config.json (written at train time). "
            "Only use this if config.json is missing and you trained with "
            "--action_scale_mult (e.g. 0.5)."
        ),
    )
    return parser.parse_args()


def make_cube_provider(args: argparse.Namespace) -> CubePoseProvider:
    if args.cube_pose == "mock":
        return MockCubePoseProvider()
    if args.cube_pose == "zed":
        return ZedCubePoseProvider(serial_number=args.zed_serial)
    raise ValueError(f"unknown cube pose mode: {args.cube_pose}")


def main() -> int:
    args = parse_args()

    if args.list_ports:
        ports = list_hand_ports()
        if not ports:
            print("No Aero Hand ports found under /dev/serial/by-id/")
            return 1
        print("\n".join(ports))
        return 0

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    checkpoint = resolve_checkpoint(args.checkpoint, env_name=args.env_name)
    print(f"Checkpoint: {checkpoint}")
    print(f"Env: {args.env_name}")
    print(f"JAX backend: {jax.default_backend()} devices={jax.devices()}")

    cube_provider = make_cube_provider(args)
    cube_pose = cube_provider.get_pose()
    print(
        f"Cube pose [{cube_pose.source}]: pos={cube_pose.position.tolist()} "
        f"quat_wxyz={cube_pose.quaternion_wxyz.tolist()}"
    )
    print(
        "Place the physical cube near the palm center at ~"
        f"{MOCK_CUBE_POS.tolist()} m (sim frame) before the hold ends."
    )

    jit_inference_fn = load_policy(
        checkpoint, seed=args.seed, env_name=args.env_name
    )

    try:
        cmd_bias = np.asarray(
            [float(x) for x in str(args.cmd_bias).split(",")],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid --cmd-bias: {exc}") from exc
    if cmd_bias.shape != (7,):
        raise SystemExit(
            f"--cmd-bias must have 7 values, got {cmd_bias.shape[0]}"
        )
    print(f"cmd_bias (sim units)={cmd_bias.tolist()}")

    hand = None
    if not args.dry_run:
        try:
            from aero_open_sdk.aero_hand import AeroHand
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            raise SystemExit(
                f"failed to import AeroHand ({missing}).\n"
                "The infer wrapper installs pyserial into the playground uv env;\n"
                "re-run via ./scripts/infer_aero_hand_rl.sh, or:\n"
                "  cd sim_rl/mujoco_playground && uv pip install pyserial\n"
                "  pip install -e sdk\n"
                "Or use --dry-run to test without hardware."
            ) from exc
        hand = AeroHand(port=args.port)
        print(f"Connected hand on {hand.ser.port}")
        if args.calibrate:
            print(
                "Running on-board homing / calibration. "
                "Keep fingers clear; this can take a few minutes..."
            )
            try:
                hand.send_homing(timeout_s=args.calibrate_timeout)
            except Exception as exc:
                raise SystemExit(f"Calibration / homing failed: {exc}") from exc
            print("Calibration complete (firmware settle = full extend / open).")
            if args.verify_open > 0:
                print(
                    f"Verifying open-palm (0°) for {args.verify_open:.1f}s — "
                    "all fingers should be fully relaxed here. If the pinky "
                    "stays bent at this step, re-home or TRIM channel 6; "
                    "do not confuse this with the later sim-home curl."
                )
                hand.set_actuations(OPEN_PALM_ACTUATION)
                time.sleep(args.verify_open)
        else:
            print("Skipping calibration (--no-calibrate).")

    default_ctrl = DEFAULT_CTRL.copy()
    action_scale = load_action_scale(
        checkpoint,
        env_name=args.env_name,
        action_scale_mult=args.action_scale_mult,
    )
    home_ctrl = default_ctrl + cmd_bias
    home_actuation = sim_array_to_actuation_array(home_ctrl)
    last_action = np.zeros(7, dtype=np.float32)
    # Policy proprio seed: last commanded sim ctrl (pre-bias), matching training.
    commanded_sim_ctrl = default_ctrl.copy()
    commanded_actuation = list(home_actuation)
    rng = jax.random.PRNGKey(args.seed)

    print(
        f"Moving to sim home ctrl={default_ctrl.tolist()} "
        f"+ bias -> exec={home_ctrl.tolist()} "
        f"-> actuation(deg)={np.round(home_actuation, 2).tolist()}"
    )
    print(
        "Note: sim home curls MCP joints (~74° after settle; keyframe qpos is "
        "stale). Index uses measured real-hand slack + MCP calibration so its "
        "policy band matches sim (~17–39° MCP). Default cmd_bias is middle/ring "
        "coupling + thumb tip spacing. Pinky looking bent at this pose is "
        "expected; full open was the verify-open / post-homing extend step above."
    )
    if hand is not None:
        hand.set_actuations(home_actuation)
    print(f"Holding home for {args.start_hold:.1f}s (place cube now)...")
    time.sleep(args.start_hold)

    if args.vel <= 0:
        raise SystemExit(f"--vel must be > 0, got {args.vel}")
    # Policy step count from sim-time duration; wall period stretched by 1/vel.
    n_steps = max(1, int(args.duration / args.dt))
    wall_dt = args.dt / args.vel
    wall_duration = args.duration / args.vel
    print(
        f"Running policy for {args.duration:.1f}s policy-time "
        f"({n_steps} steps @ dt={args.dt}s, vel={args.vel:g} → "
        f"wall {wall_dt:.3f}s/step, ~{wall_duration:.1f}s wall-clock)"
    )
    try:
        for step_i in range(n_steps):
            t0 = time.perf_counter()
            cube_pose = cube_provider.get_pose()

            if args.proprio_from_actuators:
                if hand is not None:
                    read = hand.get_actuations()
                    if read is None:
                        print("warn: get_actuations failed; using last command")
                        read = commanded_actuation
                else:
                    read = commanded_actuation
                sim_proprio = (
                    np.asarray(
                        actuation_array_to_sim_array(read), dtype=np.float32
                    )
                    - cmd_bias
                )
            else:
                # Match training proprio_source=ctrl: commanded motor targets.
                sim_proprio = commanded_sim_ctrl

            obs = build_obs(sim_proprio, last_action)
            rng, act_rng = jax.random.split(rng)
            action = np.asarray(
                jit_inference_fn(obs, act_rng)[0], dtype=np.float32
            ).ravel()

            sim_cmd = default_ctrl + action * action_scale
            commanded_sim_ctrl = sim_cmd.astype(np.float32)
            exec_cmd = sim_cmd + cmd_bias
            commanded_actuation = sim_array_to_actuation_array(exec_cmd)
            last_action = action

            if hand is not None:
                hand.set_actuations(commanded_actuation)

            if step_i % 20 == 0:
                print(
                    f"step={step_i:4d} "
                    f"cube={np.round(cube_pose.position, 3).tolist()} "
                    f"action={np.round(action, 3).tolist()} "
                    f"sim_cmd={np.round(sim_cmd, 4).tolist()} "
                    f"exec={np.round(exec_cmd, 4).tolist()} "
                    f"act={np.round(commanded_actuation, 1).tolist()}"
                )

            elapsed = time.perf_counter() - t0
            sleep_s = wall_dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if hand is not None and args.open_on_exit:
            print("Opening palm...")
            hand.set_actuations(OPEN_PALM_ACTUATION)
            time.sleep(0.5)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
