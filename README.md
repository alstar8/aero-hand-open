# Scripts

## MuJoCo RL environment

CUDA 12 JAX (recommended):

```bash
./scripts/setup_mujoco_rl_env.sh
```

## MuJoCo RL training

Train the PPO baseline (default env: `AeroCubeRotateZAxis`):

```bash
./scripts/train_mujoco_rl_baseline.sh
```

Common options:

```bash
./scripts/train_mujoco_rl_baseline.sh --gpu 0
./scripts/train_mujoco_rl_baseline.sh --smoke
./scripts/train_mujoco_rl_baseline.sh --env_name AeroCubeRotateZAxis
./scripts/train_mujoco_rl_baseline.sh --suffix my-run
./scripts/train_mujoco_rl_baseline.sh --use_tb
./scripts/train_mujoco_rl_baseline.sh --use_wandb
./scripts/train_mujoco_rl_baseline.sh --domain_randomization
./scripts/train_mujoco_rl_baseline.sh -- --use_tb --domain_randomization
```

Play a checkpoint:

```bash
./scripts/train_mujoco_rl_baseline.sh --play_only --load_checkpoint_path PATH
./scripts/train_mujoco_rl_baseline.sh --play_only --load_checkpoint_path latest
./scripts/train_mujoco_rl_baseline.sh --play_only --load_checkpoint_path sim_rl/mujoco_playground/logs/AeroCubeRotateZAxis-20260727-194230-baseline/checkpoints/000066846720
```

## Real-hand RL inference

Run a trained `AeroCubeRotateZAxis` policy on the Aero Hand Open. Moves to the sim home pose, then closes the control loop. Cube pose: `--cube-pose mock` (hardcoded near training reset) or `--cube-pose zed` (stub).

```bash

# Real hand
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint latest --gpu 1 --duration 30
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint sim_rl/mujoco_playground/logs/AeroCubeRotateZAxis-20260727-194230-baseline/checkpoints/000066846720 --gpu 0 --duration 30
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint PATH --gpu 1
./scripts/infer_aero_hand_rl.sh --cube-pose zed --checkpoint latest --gpu 1
./scripts/infer_aero_hand_rl.sh --list-ports
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint latest --port /dev/ttyACM0
./scripts/infer_aero_hand_rl.sh --cube-pose mock --checkpoint latest --open-on-exit
```

## Hand dexterity demo

Install the SDK (once):

```bash
pip install -e sdk
```

```bash
./scripts/demo_hand_dexterity.py
./scripts/demo_hand_dexterity.py --list-ports
./scripts/demo_hand_dexterity.py --port /dev/ttyACM0
./scripts/demo_hand_dexterity.py --skip-poses
./scripts/demo_hand_dexterity.py --repeat 2
./scripts/demo_hand_dexterity.py --hold 1.0
```

