# Aero Hand VR teleoperation

WebXR (Meta Quest) teleop for the bare Aero Hand Open: track the
operator's right hand, calibrate finger range of motion, then stream 
to the real hand over the SDK.

## Quick start

```bash
cd teleoperation
uv sync
```

Generate a self-signed TLS cert (WebXR requires HTTPS on LAN, or plain
HTTP on `localhost`):

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem \
  -out certs/cert.pem -days 365 -nodes -subj "/CN=$(hostname -I | awk '{print $1}')"
```

Run the server:

```bash
uv run python -m server --cert certs/cert.pem --key certs/key.pem
```

Open `https://<your-LAN-ip>:8000` in the **Meta Quest Browser** (not on a
PC). Accept the self-signed cert warning. Tap **Enter VR**.

There is no desktop video stream — the headset *is* the client. Opening
the page in Chrome/Firefox on your laptop will show
`WebXR unavailable` / `XR start failed`.

## Flow

1. **Finger calibration** -- a head-locked panel walks through 6 poses.
   Pinch your **left** thumb and index finger together to advance each
   step. Bare-hand gesture on purpose: no controller is needed at all,
   so nothing displaces WebXR hand tracking for either hand.
2. **Streaming** -- once calibration completes, every tracked right-hand
   sample is normalised against the calibration and sent straight to the
   real Aero Hand (`server/hand_driver.py`, over `aero_open_sdk`). No
   engage step -- it just starts.
3. **Quit** -- Ctrl+C. The driver parks the hand open before releasing
   the serial port.

## Troubleshooting XR start failed

- **Opened on a PC browser.** Use the Quest headset browser at
  `https://10.x.x.x:8000` (your LAN IP). Desktop browsers have no
  immersive WebXR session for this app.
- **Wrong URL / cert mismatch.** Cert CN must match the IP you type.
  Regenerate with that IP in `-subj "/CN=..."`.
- **HTTP instead of HTTPS on Quest.** WebXR on device requires TLS
  (`--cert` / `--key`).
- **Hand tracking off.** Quest Settings → Movement tracking → Hand
  tracking → On. Put controllers down so bare hands are tracked.
- After a failed attempt, hard-refresh the Quest page and tap Enter VR
  again; the on-page error text now shows the browser's reason.

## CLI flags

| flag | default | meaning |
|---|---|---|
| `--hand-port` | auto-detect | Serial port for the Aero Hand |
| `--port` | `8000` | HTTP/HTTPS port |
| `--cert` / `--key` | -- | TLS cert + key (required for non-localhost Quest) |

## Layout

```
teleoperation/
  pyproject.toml        uv project; depends on the local sdk/ (path source)
  server/
    __main__.py          CLI entry point
    server.py             aiohttp app: WS handling, calibration wiring
    hand_driver.py        AeroHandDriver -- wraps aero_open_sdk.AeroHand
    calibration.py         FingerCalibrationFSM (pure, no I/O)
    messages.py            WS wire format (hand/button in, phase/prompt out)
  webxr_app/static/       Three.js WebXR frontend (adapted from
                           vr_arm_teleop, wrist/workspace/point-cloud
                           machinery stripped out since there's no arm
                           or camera rig here)
```

