# RL Inference on Real Hardware

Real-hardware inference harness for RL policies trained on the Aero Hand with
[`mujoco_playground`](../sim_rl/mujoco_playground) (the `AeroCubeRotateZAxis`
task: rotating a cube about the Z axis using 5-finger tendon-driven grasp).
Includes a validated PPO checkpoint, the on-hand inference script, calibration
tooling, and sim-vs-real comparison scripts.

This module is a standalone [`uv`](https://docs.astral.sh/uv/) project inside
the monorepo — it depends on [`../sdk`](../sdk) (`aero-open-sdk`) and
[`../sim_rl/mujoco_playground`](../sim_rl/mujoco_playground) as local editable
packages, so changes to either are picked up immediately without reinstalling.

## Setup

```bash
# From the repo root, on branch ppo-aero-hand:
git submodule sync && git submodule update --init --recursive

# The aero_hand env (sim_rl/mujoco_playground) needs a few local fixes that
# aren't upstreamed to google-deepmind/mujoco_playground -- see "Why the
# patch exists" below. Apply it once per submodule checkout:
git apply -p1 --directory=sim_rl/mujoco_playground rl_inference/patches/aero_hand_env.patch

cd rl_inference
uv sync
```

`pyzed` (the ZED camera SDK's Python bindings, used only for human-visible
cube-tracking overlay in `run_ppo_on_hand.py` -- **it does not feed the
policy's observations**) is not on PyPI and can't be a `uv` dependency. Install
the ZED SDK for your platform, then run its bundled installer script into this
project's venv:

```bash
uv run /usr/local/zed/get_python_api.py   # path depends on your ZED SDK install
```

If you don't have a ZED camera connected, `run_ppo_on_hand.py` will still run
the policy on the hand -- cube tracking is just for your own visual reference,
the observation the policy sees is built entirely from `hand.get_actuations()`.

### Why the patch exists

The upstream `google-deepmind/mujoco_playground` submodule doesn't know about
our hand. The bundled `AeroCubeRotateZAxis` env under
`_src/manipulation/aero_hand/` needed local fixes to be trainable and to
transfer to the real 38mm/27.4g cube:

- `impl="jax"` set explicitly in `default_config()` (a locked-`ConfigDict` key
  error otherwise).
- Cube geometry/mass corrected to the real cube: 38mm edge, 27.4g (pine
  density × volume), was 50mm/69.2g.
- A small default pinky-flexion bias (`_default_tendon[3] += 0.015`) and a
  light `torques` reward penalty (`-0.0001`) -- real-hand testing showed a
  single finger (pinky or middle, depending on checkpoint) locking into a
  rigid, static hold that let the cube slip. **Earlier real-hand runs in this
  session were accidentally using a stale, unpatched copy of this env from a
  pip-installed package** (not this submodule) that was missing the pinky
  bias -- if you're comparing against old logs, keep that in mind.

`rl_inference/patches/aero_hand_env.patch` is the full diff; apply it after
every fresh `git submodule update`.

## Checkpoint

[`checkpoints/AeroCubeRotateZAxis-cube38mm_BEST/`](checkpoints/AeroCubeRotateZAxis-cube38mm_BEST)
-- PPO, trained on the corrected 38mm/27.4g cube, sim reward 68.9. This is the
best checkpoint validated on real hardware so far: with the kinematic-pose
calibration below and `--max_actuation_step_deg=10.0`, 0 clipping across a
full 300-step run and real cube rotation observed (not just a static hold).
`rollout0.mp4` in that directory is a sim rollout recording.

Note the trained policy is not perfectly reliable even in sim -- across 20
random seeds only 6/20 completed a 300-step rollout without dropping the cube
(see `scripts/log_sim_rollout.py`). Don't assume every real-hand drop is a
sim-to-real gap; some of it is the policy's own ceiling.

## Calibration before running

The policy commands are deltas from a reference pose -- if that reference is
the hand's raw mechanical zero, tendons have no travel room in the
"loosen further" direction and every step clips. `scripts/set_sim_joint_pose.py`
fixes this by taking the simulator's own `default_pose` (the 16 joint angles
from the XML `"home"` keyframe) and converting it into real actuator degrees
via the SDK's actual joint→actuation kinematic model
(`aero_open_sdk.joints_to_actuations.JointsToActuationsModel.hand_actuations()`),
plus a small manual thumb adjustment (`thumb_cmc_abd +15°`, `thumb_cmc_flex
+15°`) found to work best on top of that pose.

```bash
cd rl_inference
uv run scripts/set_sim_joint_pose.py            # homes, then ramps to the pose (real motor movement)
uv run scripts/set_sim_joint_pose.py --skip_homing   # skip homing, ramp from current pose
```

Anchor drifts after every run, so recalibrate before each fresh policy run.

An older, cruder alternative (fixed normalized neutral values instead of the
sim's actual pose) is still available via `run_ppo_on_hand.py
--move_to_neutral=true --neutral_finger_deg=144 --neutral_thumb_deg=60`, kept
for reference; prefer `set_sim_joint_pose.py`.

## Running the policy

Calibration and policy execution are always two separate invocations —
`--dry_run=true` gates `--move_to_neutral` but **not** `--do_homing`, so
combining calibration and a dry-run policy test in one call can silently home
the hand for real while only logging the "move to neutral" step.

```bash
cd rl_inference

# Dry run first: computes and logs commands, sends nothing to the motors.
uv run scripts/run_ppo_on_hand.py \
  --checkpoint_dir checkpoints/AeroCubeRotateZAxis-cube38mm_BEST/checkpoints \
  --dry_run=true \
  --do_homing=false \
  --move_to_neutral=false \
  --max_steps=300 \
  --debug_every=10 \
  --max_actuation_step_deg=10.0

# Live, once the dry run looks sane:
uv run scripts/run_ppo_on_hand.py \
  --checkpoint_dir checkpoints/AeroCubeRotateZAxis-cube38mm_BEST/checkpoints \
  --dry_run=false \
  --do_homing=false \
  --move_to_neutral=false \
  --max_steps=300 \
  --debug_every=10 \
  --max_actuation_step_deg=10.0
```

(`--checkpoint_step` is optional -- defaults to the latest step in the
directory.)

### Useful flags for tuning

| Flag | Effect |
|---|---|
| `--max_actuation_step_deg` (default `5.0`) | Per-step slew-rate clamp, the main safety/smoothness knob. `5` is safe but flat; `10-15` gave livelier, still-safe rotation on `cube38mm_BEST`; `50` was too jerky and lost the cube once. |
| `--actuation_gain_deg` (default `1.0`) | Global scale from sim action units to real degrees. |
| `--thumb_gain` / `--finger_gain` (default `1.0` each) | Per-group multiplier on top of the global gain -- thumb channels (0-2) vs finger tendons (3-6). Useful for fixing a specific finger being too rigid/too weak without touching the others. |
| `--actuation_bias_deg` (default `0.0`) | Constant offset added after scaling; a cruder alternative to `set_sim_joint_pose.py`. |
| `--servo_speed` / `--servo_torque` | Direct servo dynamics (0-32766 / 0-1000). Not yet tuned in this session -- `--servo_torque` is the most direct lever for "hold looser." |
| `--match_homing_dynamics` | If true, applies homing-like (typically slower/softer) servo speed/torque before running instead of whatever was set previously. |
| `--use_hand_state` (default `true`) | Build the observation from real actuations; `false` feeds zeros (debug only). |
| `--rate_hz` | Control loop rate; `<=0` uses the env's own `ctrl_dt`. |

## Sim-vs-real comparison

- `scripts/diagnose_sim_real_obs.py` -- read-only, no motor commands. Prints
  per-channel overlap % between the sim action range and what the real
  hardware can physically reach from its current pose.
- `scripts/log_sim_rollout.py` -- runs a checkpoint in sim and logs
  step/action/motor_target/reward/cube-pose to CSV.
- `scripts/parse_real_log.py <real_log_file> <out.csv>` -- parses a
  `run_ppo_on_hand.py --debug_every=1` log into the same CSV shape, so the two
  can be diffed channel-by-channel (mean/min/max per actuator) to see which
  channel is most desynced between sim and real.
