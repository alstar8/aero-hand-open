"""Take the simulator's default_pose (16 joint angles from the XML 'home'
keyframe, radians) and set the equivalent pose on the real Aero Hand, using
the SDK's actual joint->actuation kinematic model (tendon coupling
coefficients) rather than an ad-hoc guessed normalized target.

Sim JOINT_NAMES order: [index(3), middle(3), ring(3), pinky(3), thumb(4)]
SDK hand_actuations() order:   [thumb(4), index(3), middle(3), ring(3), pinky(3)]
"""
import argparse
import sys
import time
import numpy as np

HOMING_TIMEOUT_S = 175.0
RAMP_DURATION_S = 3.0
DT = 0.05


def sim_to_sdk_joint_order(sim16):
  sim16 = np.asarray(sim16, dtype=np.float64)
  index_, middle, ring, pinky, thumb = (
      sim16[0:3], sim16[3:6], sim16[6:9], sim16[9:12], sim16[12:16]
  )
  return np.concatenate([thumb, index_, middle, ring, pinky])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--skip_homing", action="store_true")
  args = parser.parse_args()

  import os
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  from mujoco_playground import registry
  from aero_open_sdk.aero_hand import AeroHand

  print("Loading sim default_pose (AeroCubeRotateZAxis)...")
  env = registry.load("AeroCubeRotateZAxis")
  default_pose_rad = np.array(env._default_pose, dtype=np.float64)
  default_pose_deg_sim_order = np.degrees(default_pose_rad)
  joint_positions_sdk_deg = sim_to_sdk_joint_order(default_pose_deg_sim_order)

  # User-requested adjustments on top of the sim's own default_pose. SDK
  # joint order is [thumb_cmc_abd, thumb_cmc_flex, thumb_mcp, thumb_ip,
  # index(3), ...], so thumb_cmc_abd=index 0, thumb_cmc_flex=index 1.
  THUMB_ABD_OFFSET_DEG = 0.0   # was +30, lowered back by 15
  THUMB_FLEX_OFFSET_DEG = 30.0  # new: flexion joint, +15
  joint_positions_sdk_deg[0] += THUMB_ABD_OFFSET_DEG
  joint_positions_sdk_deg[1] += THUMB_FLEX_OFFSET_DEG
  print(f"Applied thumb_cmc_abd offset: +{THUMB_ABD_OFFSET_DEG} deg "
        f"-> {joint_positions_sdk_deg[0]:.3f} deg")
  print(f"Applied thumb_cmc_flex offset: +{THUMB_FLEX_OFFSET_DEG} deg "
        f"-> {joint_positions_sdk_deg[1]:.3f} deg")
  print("Sim default_pose, SDK joint order (deg):")
  print(np.round(joint_positions_sdk_deg, 3))

  print("\nConnecting to Aero Hand...")
  hand = AeroHand(port=None)

  # Clamp to the hand's real joint-angle limits (same as set_joint_positions does).
  jl = np.asarray(hand.joint_lower_limits, dtype=np.float64)
  ju = np.asarray(hand.joint_upper_limits, dtype=np.float64)
  clamped = np.clip(joint_positions_sdk_deg, jl, ju)
  if not np.allclose(clamped, joint_positions_sdk_deg, atol=1e-6):
    print("NOTE: some joint angles were outside real joint limits and got clamped:")
    print("  wanted: ", np.round(joint_positions_sdk_deg, 2))
    print("  clamped:", np.round(clamped, 2))

  # Real joint->actuation kinematic model (tendon coupling coefficients).
  target_actuation_deg = np.array(
      hand.joints_to_actuations_model.hand_actuations(clamped.tolist()),
      dtype=np.float64,
  )
  al = np.asarray(hand.actuation_lower_limits, dtype=np.float64)
  au = np.asarray(hand.actuation_upper_limits, dtype=np.float64)
  target_actuation_deg_clipped = np.clip(target_actuation_deg, al, au)
  print("\nTarget actuator degrees (from joint kinematics):")
  print("  raw:     ", np.round(target_actuation_deg, 3))
  print("  clipped: ", np.round(target_actuation_deg_clipped, 3))
  print(f"  hw limits: lower={np.round(al,3)} upper={np.round(au,3)}")

  if args.skip_homing:
    print("\n--skip_homing set: ramping directly from current pose.")
  else:
    print(f"\nSending homing command (up to {HOMING_TIMEOUT_S:.0f}s)...")
    hand.send_homing(timeout_s=HOMING_TIMEOUT_S)

  start = hand.get_actuations()
  start = np.zeros(7, dtype=np.float64) if start is None else np.asarray(start, dtype=np.float64)
  print(f"\nCurrent actuations (deg): {np.round(start, 3)}")

  ramp_steps = max(1, int(RAMP_DURATION_S / DT))
  print(f"Ramping to sim-pose target over {RAMP_DURATION_S:.1f}s ({ramp_steps} steps)...")
  for i in range(ramp_steps):
    t = (i + 1) / ramp_steps
    cmd = start + t * (target_actuation_deg_clipped - start)
    hand.set_actuations(cmd.tolist())
    time.sleep(DT)

  time.sleep(0.3)
  final = hand.get_actuations()
  final = np.asarray(final, dtype=np.float64) if final is not None else None
  print(f"\nFinal actuations (deg): {np.round(final, 3) if final is not None else 'READ FAILED'}")
  if final is not None:
    dev = np.max(np.abs(final - target_actuation_deg_clipped))
    print(f"Max deviation from target: {dev:.2f} deg")

  hand.close()
  print("\nDone. Hand left in the sim's default_pose (converted via real joint "
        "kinematics). Run the policy next with --do_homing=false "
        "--move_to_neutral=false to reuse it as reference.")


if __name__ == "__main__":
  sys.exit(main())
