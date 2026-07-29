#!/usr/bin/env python3
"""Sim vs real observation diagnostic for the Aero Hand (read-only, no motor commands).

Prints, once, a static table per actuator channel showing:
  - sim_default:      default_tendon from the MJX 'home' keyframe (abs, m/rad)
  - sim_action_range: [default - action_scale, default + action_scale] -- the
                       full range a trained policy can command in sim-space
  - hw_range (deg):   the hand's physical actuation limits, raw SDK degrees
  - hw_range (sim):   the same hw limits converted into sim units (m/rad),
                       anchored at the hand's CURRENT pose as the zero-delta
                       reference (same anchoring run_ppo_on_hand.py uses)
  - overlap:          what fraction of sim_action_range actually fits inside
                       hw_range (sim) -- <100% means the policy can command
                       values the real hardware physically cannot reach on
                       that channel, which is what caused the ring/pinky
                       clipping observed with checkpoint 267386880.

Then loops at low rate printing only the *current* real reading (raw degrees
and reconstructed sim-units value) per channel, so you can move the fingers
by hand and watch where they sit relative to the static ranges above.

No motor commands are ever sent -- this only calls hand.get_actuations().

Usage (from rl_inference/):
    uv run scripts/diagnose_sim_real_obs.py --env_name AeroCubeRotateZAxis
    uv run scripts/diagnose_sim_real_obs.py --port /dev/ttyACM0 --rate_hz 2 --duration_s 30
"""

import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Sim actuator order == hardware SDK order comment, copied verbatim from
# run_ppo_on_hand.py::_convert_hand_actuations_to_sim_sensors to keep the
# channel mapping identical between the two scripts.
#   Hardware (SDK) actuation order (degrees):
#     [thumb_cmc_abd, thumb_cmc_flex, thumb_tendon, index, middle, ring, pinky]
#   Sim actuator order (motor_targets / default_tendon / action_scale):
#     [index, middle, ring, pinky, thumb_cmc_abd, th1_tendon, th2_tendon]
SIM_CHANNEL_NAMES = [
    "index_tendon", "middle_tendon", "ring_tendon", "pinky_tendon",
    "thumb_cmc_abd", "th1_tendon", "th2_tendon",
]
SIM_UNITS = ["m", "m", "m", "m", "rad", "m", "m"]

_DEG_TO_RAD = np.pi / 180.0
_RAD_TO_DEG = 180.0 / np.pi


def _convert_hand_actuations_to_sim_sensors(
    *,
    actuations_deg: np.ndarray,
    ref_actuations_deg: np.ndarray,
    default_tendon_sim: np.ndarray,
    motor_pulley_radius_mm: float,
) -> np.ndarray:
  """Hardware degrees -> sim-space absolute value, in SIM_CHANNEL_NAMES order.

  Kept numerically identical to run_ppo_on_hand.py's version of this
  function (including the hardware<->sim index mapping) -- do not let the
  two drift apart, or this diagnostic stops meaning anything.
  """
  actuations_deg = np.asarray(actuations_deg, dtype=np.float64)
  ref_actuations_deg = np.asarray(ref_actuations_deg, dtype=np.float64)
  default_tendon_sim = np.asarray(default_tendon_sim, dtype=np.float64)

  thumb_abd = (
      default_tendon_sim[4]
      + (actuations_deg[0] - ref_actuations_deg[0]) * _DEG_TO_RAD
  )

  radius_m = float(motor_pulley_radius_mm) / 1000.0

  def _delta_len_m(delta_deg: float) -> float:
    return float(delta_deg) * _DEG_TO_RAD * radius_m

  idx_if = default_tendon_sim[0] + _delta_len_m(actuations_deg[3] - ref_actuations_deg[3])
  idx_mf = default_tendon_sim[1] + _delta_len_m(actuations_deg[4] - ref_actuations_deg[4])
  idx_rf = default_tendon_sim[2] + _delta_len_m(actuations_deg[5] - ref_actuations_deg[5])
  idx_pf = default_tendon_sim[3] + _delta_len_m(actuations_deg[6] - ref_actuations_deg[6])
  idx_th1 = default_tendon_sim[5] + _delta_len_m(actuations_deg[1] - ref_actuations_deg[1])
  idx_th2 = default_tendon_sim[6] + _delta_len_m(actuations_deg[2] - ref_actuations_deg[2])

  # Reassemble into SIM_CHANNEL_NAMES order: [index, middle, ring, pinky, thumb_abd, th1, th2]
  return np.array([idx_if, idx_mf, idx_rf, idx_pf, thumb_abd, idx_th1, idx_th2])


def _fmt_row(cells, widths):
  return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--env_name", default="AeroCubeRotateZAxis")
  parser.add_argument("--port", default=None, help="Serial port; None = auto-detect")
  parser.add_argument("--rate_hz", type=float, default=2.0)
  parser.add_argument("--duration_s", type=float, default=0.0,
                       help="0 = run until Ctrl+C")
  args = parser.parse_args()

  import jax  # noqa: E402  (import after JAX_PLATFORMS is set)
  from mujoco_playground import registry  # noqa: E402
  from aero_open_sdk.aero_hand import AeroHand  # noqa: E402
  try:
    from aero_open_sdk.joints_to_actuations import MOTOR_PULLEY_RADIUS as PULLEY_MM
  except Exception:  # pylint: disable=broad-except
    PULLEY_MM = 9.0

  print(f"Loading env config for {args.env_name}...")
  env_cfg = registry.get_default_config(args.env_name)
  env_cfg["impl"] = "jax"
  env = registry.load(args.env_name, config=env_cfg)

  default_tendon = np.asarray(getattr(env, "_default_tendon", np.zeros(7)), dtype=np.float64)
  action_scale = np.asarray(getattr(env._config, "action_scale", [1.0] * 7), dtype=np.float64)

  sim_lo = default_tendon - action_scale
  sim_hi = default_tendon + action_scale

  print(f"Connecting to Aero Hand (port={args.port or 'auto'})...")
  hand = AeroHand(port=args.port)

  hw_lo_deg = np.asarray(hand.actuation_lower_limits, dtype=np.float64)
  hw_hi_deg = np.asarray(hand.actuation_upper_limits, dtype=np.float64)

  ref_deg = hand.get_actuations()
  if ref_deg is None:
    print("WARNING: get_actuations() timed out on first read; retrying once...")
    time.sleep(0.2)
    ref_deg = hand.get_actuations()
  if ref_deg is None:
    raise RuntimeError("Could not read current hand actuations (serial timeout).")
  ref_deg = np.asarray(ref_deg, dtype=np.float64)

  # Hardware limits converted into sim units, anchored at the CURRENT pose.
  hw_lo_sim = _convert_hand_actuations_to_sim_sensors(
      actuations_deg=hw_lo_deg, ref_actuations_deg=ref_deg,
      default_tendon_sim=default_tendon, motor_pulley_radius_mm=PULLEY_MM,
  )
  hw_hi_sim = _convert_hand_actuations_to_sim_sensors(
      actuations_deg=hw_hi_deg, ref_actuations_deg=ref_deg,
      default_tendon_sim=default_tendon, motor_pulley_radius_mm=PULLEY_MM,
  )
  # thumb_cmc_flex/thumb_tendon direction sign can flip lo/hi after conversion;
  # normalize so lo <= hi per channel before computing overlap.
  hw_lo_sim, hw_hi_sim = np.minimum(hw_lo_sim, hw_hi_sim), np.maximum(hw_lo_sim, hw_hi_sim)

  print()
  print("=== STATIC RANGE COMPARISON (anchored at current hand pose as zero-delta ref) ===")
  print(f"Reference actuations (deg, SDK order thumb_abd/flex/tendon,idx,mid,ring,pinky): {np.round(ref_deg, 2)}")
  print()
  headers = ["channel", "unit", "sim_default", "sim_range", "hw_range(sim)", "overlap%", "flag"]
  widths = [15, 5, 12, 24, 24, 9, 4]
  print(_fmt_row(headers, widths))
  print("-" * (sum(widths) + 2 * (len(widths) - 1)))

  for i, name in enumerate(SIM_CHANNEL_NAMES):
    unit = SIM_UNITS[i]
    s_lo, s_hi = sim_lo[i], sim_hi[i]
    h_lo, h_hi = hw_lo_sim[i], hw_hi_sim[i]
    inter_lo, inter_hi = max(s_lo, h_lo), min(s_hi, h_hi)
    inter_len = max(0.0, inter_hi - inter_lo)
    sim_len = max(1e-12, s_hi - s_lo)
    overlap_pct = 100.0 * inter_len / sim_len
    flag = "OK" if overlap_pct > 99.9 else "!!"
    fmt = "%.4f" if unit == "m" else "%.3f"
    print(_fmt_row([
        name, unit,
        fmt % default_tendon[i],
        f"[{fmt % s_lo}, {fmt % s_hi}]",
        f"[{fmt % h_lo}, {fmt % h_hi}]",
        f"{overlap_pct:5.1f}",
        flag,
    ], widths))

  print()
  print("overlap% < 100 means the policy's sim action range extends outside what")
  print("the real hardware can physically reach on that channel (anchored at the")
  print("current pose) -- expect systematic clipping there regardless of gain tuning.")
  print()
  print("=== LIVE READING (no commands sent; move fingers by hand to explore) ===")
  print("Ctrl+C to stop.")
  print()

  dt = 1.0 / max(args.rate_hz, 1e-6)
  t_start = time.monotonic()
  live_headers = ["channel", "raw_deg", "sim_value", "in_sim_range?"]
  live_widths = [15, 10, 12, 14]
  try:
    while args.duration_s <= 0 or (time.monotonic() - t_start) < args.duration_s:
      cur_deg = hand.get_actuations()
      if cur_deg is None:
        print("  [get_actuations() timed out]")
        time.sleep(dt)
        continue
      cur_deg = np.asarray(cur_deg, dtype=np.float64)
      cur_sim = _convert_hand_actuations_to_sim_sensors(
          actuations_deg=cur_deg, ref_actuations_deg=ref_deg,
          default_tendon_sim=default_tendon, motor_pulley_radius_mm=PULLEY_MM,
      )
      print(_fmt_row(live_headers, live_widths))
      for i, name in enumerate(SIM_CHANNEL_NAMES):
        unit = SIM_UNITS[i]
        fmt = "%.4f" if unit == "m" else "%.3f"
        in_range = sim_lo[i] <= cur_sim[i] <= sim_hi[i]
        print(_fmt_row([
            name,
            "%.2f" % cur_deg[i],
            fmt % cur_sim[i],
            "yes" if in_range else "NO (outside sim range)",
        ], live_widths))
      print()
      time.sleep(dt)
  except KeyboardInterrupt:
    print("\n[stopped by user]")
  finally:
    hand.close()


if __name__ == "__main__":
  sys.exit(main())
