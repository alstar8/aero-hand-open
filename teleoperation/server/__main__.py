"""VR (WebXR) finger teleoperation for the bare Aero Hand.

Quick start:

    mkdir -p certs
    openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem \\
      -out certs/cert.pem -days 365 -nodes -subj "/CN=$(hostname -I | awk '{print $1}')"

    uv run python -m server --cert certs/cert.pem --key certs/key.pem

Open the WebXR page at https://<your-LAN-ip>:8000 in the Quest browser.
Calibration: pinch left thumb + index to advance each of the 6 steps.
Once calibration completes, the right hand's finger curls + thumb
abduction stream straight to the real Aero Hand -- no engage step, no
wrist tracking, nothing else. Ctrl+C to quit (parks the hand open first).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .hand_driver import AeroHandDriver
from .server import ServerConfig, TeleopServer


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hand-port", default=None,
                    help="Serial port for the Aero Hand (auto-detect if omitted)")
    ap.add_argument("--port", type=int, default=8000, help="HTTP/HTTPS port")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--cert", type=Path, default=None,
                    help="TLS cert (required for non-localhost Quest)")
    ap.add_argument("--key", type=Path, default=None, help="TLS key")
    return ap.parse_args()


async def main() -> None:
    args = _parse_args()
    hand = AeroHandDriver(port=args.hand_port)
    server = TeleopServer(
        hand_driver=hand,
        config=ServerConfig(
            host=args.host,
            port=args.port,
            cert=args.cert,
            key=args.key,
        ),
    )
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
