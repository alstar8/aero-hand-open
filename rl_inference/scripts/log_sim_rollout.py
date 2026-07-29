"""Run the trained policy in sim and log per-step obs/action/motor_targets
to CSV, in a format comparable to a real-hand debug_every=1 log, for later
sim-vs-real comparison.
"""
import argparse
import csv
import json
import pathlib
import numpy as np
import jax
import jax.numpy as jp
from etils import epath
from ml_collections import config_dict

from mujoco_playground import registry

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_CKPT_DIR = (
    _REPO_ROOT / "checkpoints" / "AeroCubeRotateZAxis-cube38mm_BEST"
    / "checkpoints" / "000300810240"
)
_DEFAULT_OUT_CSV = _REPO_ROOT / "logs" / "sim_rollout_log.csv"


def load_policy(ckpt_step_dir: epath.Path):
  from brax.training import checkpoint
  from brax.training.agents.ppo import networks as ppo_networks
  from brax.training import networks as base_networks

  cfg_path = ckpt_step_dir / "ppo_network_config.json"
  loaded = json.loads(cfg_path.read_text())
  nfk = loaded["network_factory_kwargs"]
  if "activation" in nfk:
    nfk["activation"] = base_networks.ACTIVATION[nfk["activation"]]
  for k in list(nfk.keys()):
    if k.endswith("kernel_init_fn") and nfk[k] is not None:
      nfk[k] = base_networks.KERNEL_INITIALIZER[nfk[k]]
  ckpt_cfg = config_dict.create(**loaded)
  networks = checkpoint.get_network(ckpt_cfg, ppo_networks.make_ppo_networks)
  make_policy = ppo_networks.make_inference_fn(networks)
  params = checkpoint.load(ckpt_step_dir)
  inference_fn = make_policy(params, deterministic=True)
  return jax.jit(inference_fn)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint_step_dir", type=str, default=str(_DEFAULT_CKPT_DIR))
  parser.add_argument("--out_csv", type=str, default=str(_DEFAULT_OUT_CSV))
  parser.add_argument("--max_steps", type=int, default=300,
                       help="Should match the real run's --max_steps.")
  parser.add_argument("--seed", type=int, default=4,
                       help="seed=0 dropped the cube at step 39; seed=4 runs "
                            "300 clean steps (only ~6/20 seeds succeed for "
                            "this checkpoint -- it is not perfectly reliable "
                            "even in sim).")
  args = parser.parse_args()

  ckpt_step_dir = epath.Path(args.checkpoint_step_dir)
  out_csv = pathlib.Path(args.out_csv)
  out_csv.parent.mkdir(parents=True, exist_ok=True)

  env = registry.load("AeroCubeRotateZAxis")
  jit_inference_fn = load_policy(ckpt_step_dir)
  jit_step = jax.jit(env.step)
  print("Policy + env loaded.")

  cube_geom_id = env.mj_model.geom("cube").id
  action_scale = np.asarray(env._config.action_scale)
  default_tendon = np.asarray(env._default_tendon)

  rng = jax.random.PRNGKey(args.seed)
  rng, reset_rng = jax.random.split(rng)
  state = env.reset(reset_rng)

  header = (
      ["step"]
      + [f"state_{i}" for i in range(14)]
      + [f"action_{i}" for i in range(7)]
      + [f"motor_target_{i}" for i in range(7)]
      + ["reward", "done", "cube_x", "cube_y", "cube_z",
         "cube_qw", "cube_qx", "cube_qy", "cube_qz"]
  )

  with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for step_i in range(args.max_steps):
      rng, act_key = jax.random.split(rng)
      act = np.asarray(jit_inference_fn(state.obs, act_key)[0])
      state = jit_step(state, jp.asarray(act))

      obs_state = np.asarray(state.obs["state"])
      motor_targets = default_tendon + act * action_scale
      data = state.data
      cube_pos = np.asarray(data.geom_xpos[cube_geom_id])
      cube_body_id = env.mj_model.body("cube").id
      cube_quat = np.asarray(data.xquat[cube_body_id])

      row = (
          [step_i]
          + obs_state.tolist()
          + act.tolist()
          + motor_targets.tolist()
          + [float(state.reward), bool(state.done)]
          + cube_pos.tolist()
          + cube_quat.tolist()
      )
      writer.writerow(row)

  print(f"Wrote {args.max_steps} steps to {out_csv}")
  print("\nColumns:")
  print(" ", ", ".join(header))
  print("\nstate_0..5 = tendon lengths (m): index,middle,ring,pinky,th1,th2")
  print("state_6 = thumb_cmc_abd (rad); state_7..13 = last_act (7)")
  print("action_0..6 = policy output [-1,1], ACTUATOR_NAMES order:")
  print("  index,middle,ring,pinky,thumb_cmc_abd,th1_tendon,th2_tendon")
  print("motor_target_0..6 = default_tendon + action*action_scale (sim units, m/rad)")


if __name__ == "__main__":
  main()
