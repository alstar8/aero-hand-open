"""Real-hardware driver: Aero Hand fingers only (no arm).

Translates normalised VR signals (5 finger curls + thumb abduction, all
0..1) into the Aero Hand 7-joint compact form via the ``aero_open_sdk``
AeroHand SDK.

Aero Hand 7-joint layout (matches AeroHand.convert_seven_joints_to_sixteen):
    slot 0  thumb_cmc_abd      <- target_thumb_abduction
    slot 1  thumb_cmc_flex     <- target_finger_curls[0]  (thumb)
    slot 2  thumb_mcp (+ip)    <- target_finger_curls[0]  (thumb)
    slot 3  index_mcp          <- target_finger_curls[1]
    slot 4  middle_mcp         <- target_finger_curls[2]
    slot 5  ring_mcp           <- target_finger_curls[3]
    slot 6  pinky_mcp          <- target_finger_curls[4]
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Indices into the AeroHand 16-joint limit arrays for the 7-joint compact
# form. Matches AeroHand.convert_seven_joints_to_sixteen.
_AERO_SLOT_IDX = (0, 1, 2, 4, 7, 10, 13)

# Keep fingers away from the hard mechanical limits during teleop.
_CURL_GAIN = 0.95
_ABD_GAIN = 1.0


def _curls_to_aero7(
    curls: np.ndarray,             # (5,) normalised 0..1 [thumb, idx, mid, ring, little]
    abd_norm: float,                # normalised 0..1 thumb CMC abduction
    lower: tuple[float, ...],       # 16-joint lower limits, degrees
    upper: tuple[float, ...],       # 16-joint upper limits, degrees
) -> list[float]:
    """Map normalised VR signals to the Aero Hand 7-joint compact form (degrees)."""

    def lerp(slot_idx: int, t: float) -> float:
        lo, hi = lower[slot_idx], upper[slot_idx]
        return lo + float(np.clip(t, 0.0, 1.0)) * (hi - lo)

    c = np.asarray(curls, dtype=np.float32)
    return [
        lerp(_AERO_SLOT_IDX[0], float(abd_norm) * _ABD_GAIN),
        lerp(_AERO_SLOT_IDX[1], float(c[0]) * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[2], float(c[0]) * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[3], float(c[1]) * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[4], float(c[2]) * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[5], float(c[3]) * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[6], float(c[4]) * _CURL_GAIN),
    ]


@dataclass(frozen=True)
class HandState:
    """Snapshot of the hand at a point in time."""

    curls: np.ndarray      # (5,) normalised, last commanded value
    actuations_deg: tuple[float, ...]  # (7,) live actuator readback
    timestamp: float


class AeroHandDriver:
    """Aero Hand fingers on real hardware. No RC5 / arm involved at all.

    All blocking SDK calls are dispatched via asyncio.to_thread so the
    event loop remains free. A threading.Lock serialises concurrent SDK
    calls in case get_state/send overlap.
    """

    def __init__(self, port: Optional[str] = None) -> None:
        self._port = port
        self._hand = None   # AeroHand, set in start()
        self._lock = threading.Lock()

        self._joint_lower: Optional[tuple[float, ...]] = None
        self._joint_upper: Optional[tuple[float, ...]] = None

        self._last_curls: np.ndarray = np.zeros(5, dtype=np.float32)

    async def start(self) -> None:
        if self._hand is not None:
            return  # idempotent

        def _connect():
            from aero_open_sdk.aero_hand import AeroHand  # noqa: PLC0415

            hand = AeroHand(port=self._port) if self._port else AeroHand()
            # AeroHand() only opens the serial port -- a stale
            # /dev/serial/by-id symlink or a powered-off hand still opens
            # fine, and set_joint_positions() is a fire-and-forget write
            # that never surfaces a dead connection. Ping the hand with a
            # position query so a bad connection fails loudly here instead
            # of teleop silently running into the void.
            probe = hand.get_actuations()
            if probe is None:
                hand.close()
                raise RuntimeError(
                    "AeroHand serial port opened but the hand did not "
                    "respond to a position query -- check it is powered "
                    "on, the port is correct, and no other process has it "
                    "open."
                )
            print(
                f"[hand] AeroHand detected, actuations(deg)="
                f"{[round(v, 1) for v in probe]}",
                flush=True,
            )
            self._joint_lower = hand.joint_lower_limits
            self._joint_upper = hand.joint_upper_limits
            self._hand = hand

        await asyncio.to_thread(_connect)

    async def stop(self) -> None:
        def _disconnect():
            if self._hand is None:
                return
            try:
                with self._lock:
                    if self._joint_lower and self._joint_upper:
                        # Move to an open/safe pose before releasing.
                        open7 = _curls_to_aero7(
                            np.zeros(5, dtype=np.float32), 1.0,
                            self._joint_lower, self._joint_upper,
                        )
                    else:
                        open7 = [0.0] * 7
                    self._hand.set_joint_positions(open7)
                    time.sleep(0.1)
                    self._hand.close()
            except Exception as exc:
                print(f"[AeroHandDriver] close on stop: {exc!r}")
            self._hand = None

        await asyncio.to_thread(_disconnect)

    async def send(self, curls: np.ndarray, abduction: float) -> None:
        if self._hand is None:
            raise RuntimeError("AeroHandDriver.send called before start()")

        def _send():
            c = np.asarray(curls, dtype=np.float32)
            if c.shape[0] != 5:
                raise ValueError("curls must have length 5")
            a = float(np.clip(abduction, 0.0, 1.0))
            joints7 = _curls_to_aero7(c, a, self._joint_lower, self._joint_upper)
            with self._lock:
                self._hand.set_joint_positions(joints7)
            self._last_curls = c.copy()

        await asyncio.to_thread(_send)

    async def get_state(self) -> HandState:
        if self._hand is None:
            raise RuntimeError("AeroHandDriver.get_state called before start()")

        def _read() -> HandState:
            with self._lock:
                actuations = self._hand.get_actuations()
            return HandState(
                curls=self._last_curls.copy(),
                actuations_deg=tuple(actuations) if actuations is not None else (),
                timestamp=time.monotonic(),
            )

        return await asyncio.to_thread(_read)
