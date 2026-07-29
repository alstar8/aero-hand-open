"""WebSocket message types (control channel only).

Deliberately minimal: this teleop drives finger curls + thumb abduction on
a bare Aero Hand, nothing else. No wrist pose, no workspace, no engage
handshake -- there is no arm here to carry a wrist through space.

Keep this file free of any business logic so the same shapes can be
referenced from the frontend (the structure is mirrored in JS).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass


# ----- Client -> Server ---------------------------------------------------

@dataclass(frozen=True)
class HandStateMsg:
    """Per-frame right-hand state. Streamed at ~30 Hz."""
    type: str = "hand"
    curls: tuple[float, ...] = (0.0,) * 5   # thumb..little, 0..1
    abduction: float = 0.0                  # raw radians (server normalizes)
    valid: bool = False


@dataclass(frozen=True)
class ButtonMsg:
    """Edge event for a digital gesture. Sent only on rising edge.

    Only ``name == 'x_click'`` is meaningful today: the client sends it on
    a left-hand thumb+index pinch, used to step through finger calibration.
    """
    type: str = "button"
    hand: str = "left"
    name: str = "x_click"
    pressed: bool = True


# ----- Server -> Client ---------------------------------------------------

@dataclass(frozen=True)
class PhaseMsg:
    """Tell the client which phase we're in, drives the UI."""
    type: str = "phase"
    phase: str = "idle"
    # 'idle' | 'finger_cal' | 'ready'


@dataclass(frozen=True)
class PromptMsg:
    """Head-locked text panel content."""
    type: str = "prompt"
    text: str | None = None
    severity: str = "info"    # 'info' | 'warn' | 'error'


# Mapping from the wire ``type`` discriminator to the inbound dataclass
# the server reconstructs. Only Client->Server messages live here; outbound
# messages are dataclasses we encode but never decode.
_CLIENT_TYPES = {
    "hand": HandStateMsg,
    "button": ButtonMsg,
}


def encode(msg) -> str:
    """Serialize any message dataclass into the JSON the client expects."""
    if not is_dataclass(msg):
        raise TypeError(f"encode expects a dataclass, got {type(msg).__name__}")
    return json.dumps(asdict(msg))


def decode(text: str):
    """Parse incoming JSON into one of the Client->Server dataclasses."""
    obj = json.loads(text)
    t = obj.get("type")
    cls = _CLIENT_TYPES.get(t)
    if cls is None:
        raise ValueError(f"unknown control message type: {t!r}")
    if cls is HandStateMsg:
        return HandStateMsg(
            curls=tuple(float(v) for v in obj.get("curls", (0.0,) * 5)),
            abduction=float(obj.get("abduction", 0.0)),
            valid=bool(obj.get("valid", False)),
        )
    if cls is ButtonMsg:
        return ButtonMsg(
            hand=str(obj.get("hand", "left")),
            name=str(obj.get("name", "x_click")),
            pressed=bool(obj.get("pressed", True)),
        )
    raise ValueError(f"unhandled control message type: {t!r}")
