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

"""Locate an Aero Hand Open on USB and run on-board homing / calibration.

Sends the HOMING command so the firmware drives each servo to contact,
calibrates offsets, and settles to the extend posture. The hand ignores
other commands while homing is in progress.

Usage:
  ./scripts/calibrate_hand.py
  ./scripts/calibrate_hand.py --port /dev/ttyACM0
  ./scripts/calibrate_hand.py --list-ports

Requires aero-open-sdk (pip install -e sdk) and a powered hand on USB.
Keep fingers clear of obstacles; the motion can take up to ~3 minutes.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
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


def list_hand_ports() -> list[str]:
    """Return candidate Aero Hand serial paths under /dev/serial/by-id/."""
    if not os.path.isdir(SERIAL_BY_ID_DIR):
        return []
    return sorted(
        os.path.join(SERIAL_BY_ID_DIR, name)
        for name in os.listdir(SERIAL_BY_ID_DIR)
        if ESP32_BY_ID_PREFIX in name
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate an Aero Hand Open connected over USB and run on-board "
            "homing / calibration."
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
        "--timeout",
        type=float,
        default=175.0,
        help="Seconds to wait for homing ACK (default: 175)",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List detected Aero Hand USB serial ports and exit",
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

    print(
        "Starting homing / calibration. Keep clear of the fingers; "
        "this can take up to a few minutes..."
    )
    print("Waiting for firmware ACK (motors may finish before the ACK arrives)...")

    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        started = time.monotonic()
        while not stop_heartbeat.wait(5.0):
            elapsed = int(time.monotonic() - started)
            print(f"  still waiting for ACK... {elapsed}s", flush=True)

    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()
    try:
        hand.send_homing(timeout_s=args.timeout)
    except KeyboardInterrupt:
        print("\nInterrupted during homing.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Homing failed: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_heartbeat.set()
        hand.close()

    print("Calibration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
