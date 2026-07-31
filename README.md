# Scripts

## MuJoCo RL environment

CUDA 12 JAX (recommended):

```bash
./scripts/setup_mujoco_rl_env.sh
```

## MuJoCo RL training

Train the PPO baseline (default env: `AeroCubeRotateZAxis`, 50mm cube).
Pass a task name positionally or with `--env_name`:

| Task | Cube |
|------|------|
| `AeroCubeRotateZAxis` | 50mm (default) |
| `AeroCubeRotateZAxis38mm` | 38mm |

Training defaults now match real deploy: proprio from commanded ctrl
(`get_actuations`-style), with domain randomization on (friction / mass /
actuator gain+zero-drift). Use `--no_domain_randomization` to turn DR off.

```bash
./scripts/train_mujoco_rl_baseline.sh
./scripts/train_mujoco_rl_baseline.sh AeroCubeRotateZAxis38mm
./scripts/train_mujoco_rl_baseline.sh --env_name AeroCubeRotateZAxis38mm
```

Cold starts compile XLA on the CPU (`ptxas`); `nvidia-smi` GPU util often
stays near 0% until the first `reward=` line, then should sit near 100%.
Compiles are cached under `~/.cache/jax` so restarts warm up much faster.
TensorBoard logging is on by default (events under the run dir); pass
`--no_tb` to disable.

Common options:

```bash
./scripts/train_mujoco_rl_baseline.sh --gpu 0
./scripts/train_mujoco_rl_baseline.sh --smoke
./scripts/train_mujoco_rl_baseline.sh AeroCubeRotateZAxis38mm --gpu 0
./scripts/train_mujoco_rl_baseline.sh --env_name AeroCubeRotateZAxis
./scripts/train_mujoco_rl_baseline.sh --suffix my-run
./scripts/train_mujoco_rl_baseline.sh --no_tb
./scripts/train_mujoco_rl_baseline.sh --use_wandb
./scripts/train_mujoco_rl_baseline.sh --no_domain_randomization
```

Play a checkpoint (writes `rollout*.mp4` into the resolved step folder):

```bash
./scripts/train_mujoco_rl_baseline.sh --play_only --load_checkpoint_path PATH
./scripts/train_mujoco_rl_baseline.sh --play_only --load_checkpoint_path latest
./scripts/train_mujoco_rl_baseline.sh --play_only --load_checkpoint_path sim_rl/mujoco_playground/logs/AeroCubeRotateZAxis-20260727-194230-baseline/checkpoints/000300810240
./scripts/train_mujoco_rl_baseline.sh AeroCubeRotateZAxis38mm --play_only --load_checkpoint_path sim_rl/mujoco_playground/logs/AeroCubeRotateZAxis38mm-20260731-113909-baseline/checkpoints/000133693440


```

## Real-hand RL inference

Run a trained policy on the Aero Hand Open. By default runs on-board
homing/calibration first, moves to the sim home pose, then closes the
control loop. Cube pose: `--cube-pose mock` (hardcoded near training reset)
or `--cube-pose zed` (stub). Skip calibration with `--no-calibrate`.

```bash

# Real hand
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint latest --gpu 1 --duration 30
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint PATH --gpu 1
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint PATH --env_name AeroCubeRotateZAxis38mm --gpu 0 --duration 60
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint PATH --no-calibrate
./scripts/infer_aero_hand_rl.sh --cube-pose zed --checkpoint latest --gpu 1
./scripts/infer_aero_hand_rl.sh --list-ports
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint latest --port /dev/ttyACM0
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint latest --open-on-exit


./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint sim_rl/mujoco_playground/logs/AeroCubeRotateZAxis38mm-20260731-113909-baseline/checkpoints/000200540160 --env_name AeroCubeRotateZAxis38mm --gpu 0 --duration 60
```

## Hand calibration & dexterity demo

Install the SDK (once):

```bash
pip install -e sdk
```

Calibrate (on-board homing — keep fingers clear; can take a few minutes):

```bash
./scripts/calibrate_hand.py
./scripts/calibrate_hand.py --list-ports
./scripts/calibrate_hand.py --port /dev/ttyACM0
```

Dexterity demo:

```bash
./scripts/demo_hand_dexterity.py
./scripts/demo_hand_dexterity.py --list-ports
./scripts/demo_hand_dexterity.py --port /dev/ttyACM0
./scripts/demo_hand_dexterity.py --skip-poses
./scripts/demo_hand_dexterity.py --repeat 2
./scripts/demo_hand_dexterity.py --hold 1.0
```

## VR teleoperation (WebXR)

Track a Meta Quest operator's right hand and stream calibrated finger
curls + thumb abduction to the real Aero Hand. Own `uv` project under
`teleoperation/` -- see [teleoperation/README.md](teleoperation/README.md)
for the full flow.

```bash
cd teleoperation
uv sync

mkdir -p certs
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem \
  -out certs/cert.pem -days 365 -nodes -subj "/CN=$(hostname -I | awk '{print $1}')"

uv run python -m server --cert certs/cert.pem --key certs/key.pem
```

Open `https://<your-LAN-ip>:8000` in the Quest browser. Pinch left thumb
+ index to advance the 6-step finger calibration; once it completes, the
tracked right hand drives the real hand directly.

