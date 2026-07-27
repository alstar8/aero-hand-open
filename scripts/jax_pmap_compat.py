"""Restore jax.device_put_replicated for brax PPO on JAX >= 0.11.

Brax 0.14.x still calls ``jax.device_put_replicated``, which JAX 0.11 removed.
Upstream brax main already switched to a NamedSharding helper; until that
lands on PyPI, this drop-in matches the JAX migration guide.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def device_put_replicated(x, devices):
  """Drop-in replacement for jax.device_put_replicated supporting pytrees."""
  mesh = Mesh(np.array(devices), ("_device_put_replicated",))
  sharding = NamedSharding(mesh, P("_device_put_replicated"))
  return jax.tree.map(
      lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding),
      x,
  )


def apply() -> bool:
  """Patch jax if needed. Returns True when a patch was applied."""
  if hasattr(jax, "device_put_replicated"):
    return False
  jax.device_put_replicated = device_put_replicated  # type: ignore[attr-defined]
  return True


apply()
