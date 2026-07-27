#!/usr/bin/env python3
"""Run playground learning/train_jax_ppo.py with JAX 0.11 brax compat."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# Headless MuJoCo rendering: must be set before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")
# Avoid mediapy / matplotlib trying to open interactive windows.
os.environ.setdefault("MPLBACKEND", "Agg")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import jax_pmap_compat  # noqa: E402, F401  # patches jax.device_put_replicated

_TRAIN = (
    _SCRIPT_DIR.parent
    / "sim_rl"
    / "mujoco_playground"
    / "learning"
    / "train_jax_ppo.py"
)
sys.argv[0] = str(_TRAIN)
runpy.run_path(str(_TRAIN), run_name="__main__")
