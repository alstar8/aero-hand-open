"""Finger range-of-motion calibration.

State machine that prompts the user through a fixed sequence of poses
(curl min/max per finger group, thumb abduction min/max) and records the
raw signal values for later normalization.

Pure logic -- no I/O. The orchestrator drives it from incoming WS events
and emits prompts back to the browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np


class CalibStepKind(Enum):
    """Whether a calibration step captures finger curls or thumb abduction."""
    CURL = "curl"
    ABDUCTION = "abd"


class CalibBound(Enum):
    """Which extreme of the calibrated range the step fills."""
    MIN = "min"
    MAX = "max"


@dataclass(frozen=True)
class CalibStep:
    """One step of the calibration script."""

    prompt: str
    kind: CalibStepKind
    bound: CalibBound
    fingers: tuple[int, ...] = ()   # which finger indices the step affects;
                                    # ignored for ABDUCTION steps


# The canonical 6-step calibration.
DEFAULT_STEPS: tuple[CalibStep, ...] = (
    CalibStep(
        prompt="Bend index, middle, ring, little fully (curled fist).\n"
               "Pinch left thumb + index to confirm.",
        kind=CalibStepKind.CURL, bound=CalibBound.MAX, fingers=(1, 2, 3, 4),
    ),
    CalibStep(
        prompt="Extend index, middle, ring, little straight.\n"
               "Pinch left thumb + index to confirm.",
        kind=CalibStepKind.CURL, bound=CalibBound.MIN, fingers=(1, 2, 3, 4),
    ),
    CalibStep(
        prompt="Bend thumb fully across the palm.\n"
               "Pinch left thumb + index to confirm.",
        kind=CalibStepKind.CURL, bound=CalibBound.MAX, fingers=(0,),
    ),
    CalibStep(
        prompt="Extend thumb straight out.\n"
               "Pinch left thumb + index to confirm.",
        kind=CalibStepKind.CURL, bound=CalibBound.MIN, fingers=(0,),
    ),
    # Bounds are swapped relative to anatomical labels: the hand's thumb
    # CMC abduction actuator goes (lower = thumb spread/up, upper = tucked
    # across palm), opposite of what "abduction min/max" suggests. Keep
    # the prompt text natural and store the captured raw values so the
    # linear remap in CalibrationRecord.apply_abduction sends the joint
    # to the right end.
    CalibStep(
        prompt="Spread thumb fully away from the palm.\n"
               "Pinch left thumb + index to confirm.",
        kind=CalibStepKind.ABDUCTION, bound=CalibBound.MIN,
    ),
    CalibStep(
        prompt="Tuck thumb in alongside the index finger.\n"
               "Pinch left thumb + index to confirm.",
        kind=CalibStepKind.ABDUCTION, bound=CalibBound.MAX,
    ),
)


@dataclass
class CalibrationRecord:
    """Captured min/max values per axis. Filled by the FSM, consumed each
    frame to remap raw signals onto [0, 1]."""

    min_curl: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))
    max_curl: np.ndarray = field(default_factory=lambda: np.ones(5, dtype=np.float32))
    min_abd: float = 0.0
    max_abd: float = float(np.pi / 2)

    def apply_curl(self, raw: np.ndarray) -> np.ndarray:
        """Remap raw curl signal to [0, 1] using the captured bounds."""
        raw = np.asarray(raw, dtype=np.float32)
        span = np.maximum(self.max_curl - self.min_curl, 1e-3)
        return np.clip((raw - self.min_curl) / span, 0.0, 1.0).astype(np.float32)

    def apply_abduction(self, raw: float) -> float:
        """Remap raw abduction to [0, 1]."""
        span = self.max_abd - self.min_abd
        if abs(span) < 1e-3:
            return 0.0
        return float(np.clip((raw - self.min_abd) / span, 0.0, 1.0))


_INITIAL_PROMPT = "Pinch left thumb + index to start finger calibration."
_DONE_PROMPT = "Calibration complete. Ready."


class FingerCalibrationFSM:
    """Walks the operator through DEFAULT_STEPS.

    External interface:
    - on_start(): begin calibration, advance to the first step's prompt.
    - on_confirm(curls, abduction): capture the current step, advance.
    - is_complete -> bool, current_prompt -> str.

    Owns a :class:`CalibrationRecord` populated as the steps complete.
    """

    def __init__(self, steps: tuple[CalibStep, ...] = DEFAULT_STEPS) -> None:
        self._steps = steps
        self._step_index = 0
        self._started = False
        self.record = CalibrationRecord()

    @property
    def is_complete(self) -> bool:
        return self._started and self._step_index >= len(self._steps)

    @property
    def current_prompt(self) -> str:
        if not self._started:
            return _INITIAL_PROMPT
        if self.is_complete:
            return _DONE_PROMPT
        return self._steps[self._step_index].prompt

    def on_start(self) -> None:
        self._started = True
        self._step_index = 0
        # Reset to defaults so a second calibration doesn't see stale bounds.
        self.record = CalibrationRecord()

    def on_confirm(self, curls: Iterable[float], abduction: float) -> None:
        """Capture the current step's reading and advance."""
        if not self._started or self.is_complete:
            return
        step = self._steps[self._step_index]
        curls_arr = np.asarray(tuple(curls), dtype=np.float32)
        if step.kind == CalibStepKind.CURL:
            target = (self.record.max_curl if step.bound == CalibBound.MAX
                      else self.record.min_curl)
            for f in step.fingers:
                target[f] = float(curls_arr[f])
        else:  # ABDUCTION
            if step.bound == CalibBound.MAX:
                self.record.max_abd = float(abduction)
            else:
                self.record.min_abd = float(abduction)
        self._step_index += 1
