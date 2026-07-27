#!/usr/bin/env python3
# Copyright 2025 TetherIA, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Locate an Aero Hand Open on USB and run a per-finger dexterity demo.

Finds the Espressif USB-JTAG serial device (or uses --port), then curls each
finger in turn and finishes with a short pose sequence.

Usage:
  ./scripts/demo_hand_dexterity.py
  ./scripts/demo_hand_dexterity.py --port /dev/ttyACM0
  ./scripts/demo_hand_dexterity.py --list-ports
  ./scripts/demo_hand_dexterity.py --repeat 2

Requires aero-open-sdk (pip install -e sdk) and a powered hand on USB.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Prefer an editable install; fall back to the in-repo SDK source tree.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SDK_SRC = _REPO_ROOT / "sdk" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

try:
    from aero_open_sdk.aero_hand import AeroHand
except ModuleNotFoundError as exc:
    raise SystemExit(
        "aero-open-sdk is not installed.\n"
        "Install it with:\n"
        "  pip install -e sdk\n"
        "or:\n"
        "  pip install aero-open-sdk\n"
    ) from exc

ESP32_BY_ID_PREFIX = "usb-Espressif_USB_JTAG_serial_debug_unit_"
SERIAL_BY_ID_DIR = "/dev/serial/by-id"

# Compact 7-DoF joint space (degrees):
#   [thumb_cmc_abd, thumb_cmc_flex, thumb_mcp, index, middle, ring, pinky]
OPEN_PALM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

FINGER_CURL = 75.0
THUMB_ABD = 80.0
THUMB_FLEX = 40.0
THUMB_MCP = 35.0

# (name, compact joint index or None for thumb multi-joint)
FINGERS = (
    ("thumb", None),
    ("index", 3),
    ("middle", 4),
    ("ring", 5),
    ("pinky", 6),
)


def list_hand_ports() -> list[str]:
    """Return candidate Aero Hand serial paths under /dev/serial/by-id/."""
    if not os.path.isdir(SERIAL_BY_ID_DIR):
        return []
    return sorted(
        os.path.join(SERIAL_BY_ID_DIR, name)
        for name in os.listdir(SERIAL_BY_ID_DIR)
        if ESP32_BY_ID_PREFIX in name
    )


def finger_pose(name: str, joint_index: int | None) -> list[float]:
    """Return a compact pose that curls one finger while leaving others open."""
    pose = list(OPEN_PALM)
    if name == "thumb":
        pose[0] = THUMB_ABD
        pose[1] = THUMB_FLEX
        pose[2] = THUMB_MCP
        return pose
    assert joint_index is not None
    pose[joint_index] = FINGER_CURL
    return pose


def hold(hand: AeroHand, pose: list[float], seconds: float) -> None:
    hand.set_joint_positions(pose)
    time.sleep(seconds)


def run_finger_wave(hand: AeroHand, hold_s: float) -> None:
    """Curl and release each finger once."""
    print("Opening palm...")
    hold(hand, OPEN_PALM, 0.8)

    for name, joint_index in FINGERS:
        print(f"  curling {name}...")
        hold(hand, finger_pose(name, joint_index), hold_s)
        print(f"  releasing {name}...")
        hold(hand, OPEN_PALM, hold_s * 0.6)


def run_pose_sequence(hand: AeroHand) -> None:
    """Short multi-finger poses to show coordinated dexterity."""
    print("Running coordinated pose sequence...")
    trajectory = [
        (OPEN_PALM, 0.4),
        # Pinch: thumb + pinky → ring → middle → index
        ([100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 0.45),
        ([100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 0.25),
        ([100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 0.45),
        ([100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 0.25),
        ([83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 0.45),
        ([83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 0.25),
        ([75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.45),
        ([75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.25),
        (OPEN_PALM, 0.5),
        # Peace
        ([90.0, 0.0, 0.0, 0.0, 0.0, 90.0, 90.0], 0.4),
        ([90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 0.5),
        ([90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 0.8),
        (OPEN_PALM, 0.5),
        # Rock
        ([0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 0.8),
        (OPEN_PALM, 0.5),
    ]
    hand.run_trajectory(trajectory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate an Aero Hand Open connected over USB and demonstrate "
            "finger dexterity."
        )
    )
    parser.add_argument(
        "--port",
        default=None,
        help=(
            "Serial port path (default: auto-detect Espressif USB-JTAG under "
            f"{SERIAL_BY_ID_DIR})"
        ),
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=921600,
        help="Serial baud rate (default: 921600)",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List detected Aero Hand USB serial ports and exit",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=0.7,
        help="Seconds to hold each curled finger (default: 0.7)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to run the full demo (default: 1)",
    )
    parser.add_argument(
        "--skip-poses",
        action="store_true",
        help="Only wave each finger; skip the coordinated pose sequence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ports = list_hand_ports()

    if args.list_ports:
        if not ports:
            print("No Aero Hand USB ports found under /dev/serial/by-id/.")
            print("Check the cable, power, and dialout permissions.")
            return 1
        print("Detected Aero Hand serial port(s):")
        for path in ports:
            print(f"  {path}")
        return 0

    if args.port is None and ports:
        print(f"Detected hand on USB: {ports[0]}")
        if len(ports) > 1:
            print(
                "Multiple hands detected; using the first. "
                "Pass --port to choose another:"
            )
            for path in ports:
                print(f"  {path}")

    try:
        hand = AeroHand(port=args.port, baudrate=args.baudrate)
    except RuntimeError as exc:
        print(f"Failed to locate/connect to Aero Hand: {exc}", file=sys.stderr)
        if ports:
            print("Candidates:", *ports, sep="\n  ", file=sys.stderr)
        else:
            print(
                "Tip: ls -l /dev/serial/by-id/   # look for Espressif USB-JTAG",
                file=sys.stderr,
            )
        return 1
    except Exception as exc:  # serial open / permission errors
        print(f"Failed to open serial port: {exc}", file=sys.stderr)
        print(
            "Tip: add your user to the dialout group, or try:\n"
            "  sudo chmod 666 /dev/ttyACM0",
            file=sys.stderr,
        )
        return 1

    print(f"Connected. Running dexterity demo ({args.repeat}x)...")
    try:
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"\n=== Pass {i + 1}/{args.repeat} ===")
            run_finger_wave(hand, hold_s=args.hold)
            if not args.skip_poses:
                run_pose_sequence(hand)
        print("Returning to open palm...")
        hold(hand, OPEN_PALM, 0.5)
    except KeyboardInterrupt:
        print("\nInterrupted; opening palm...")
        try:
            hold(hand, OPEN_PALM, 0.3)
        except Exception:
            pass
        return 130
    finally:
        hand.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
