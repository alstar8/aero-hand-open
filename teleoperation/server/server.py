"""Top-level orchestrator -- wires the AeroHandDriver into a running
aiohttp app.

Phases:
    idle         after-connect, before finger calibration
    finger_cal   walking through the FingerCalibrationFSM
    ready        calibration done; every valid right-hand sample is
                 calibrated and streamed straight to the hand

Deliberately single-loop: there is no wrist/workspace to track and no
engage handshake, so a HandStateMsg arriving over the WS is all the
"tick" this server needs -- no separate command loop.
"""

from __future__ import annotations

import asyncio
import ssl
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from .calibration import FingerCalibrationFSM
from .hand_driver import AeroHandDriver
from .messages import ButtonMsg, HandStateMsg, PhaseMsg, PromptMsg, decode, encode

_STATIC_DIR = Path(__file__).parent.parent / "webxr_app" / "static"

# Throttle "hand send failed" logging so a disconnect mid-session doesn't
# spam the console at the ~30 Hz hand-state rate.
_SEND_ERROR_LOG_PERIOD_S = 2.0


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    static_dir: Path = _STATIC_DIR
    cert: Path | None = None
    key: Path | None = None


class TeleopServer:
    """Owns one AeroHandDriver / one connection (single-operator design)."""

    def __init__(self, hand_driver: AeroHandDriver, config: ServerConfig) -> None:
        self._hand = hand_driver
        self._config = config
        self._calib = FingerCalibrationFSM()
        self._phase = "idle"
        self._latest_hand = HandStateMsg()
        self._shutdown = asyncio.Event()
        self._last_send_error_log = 0.0

    async def run(self) -> None:
        """Start the hand driver + aiohttp, block until shutdown."""
        await self._hand.start()

        app = self._make_app()
        runner = web.AppRunner(app)
        await runner.setup()

        ssl_context = None
        if self._config.cert and self._config.key:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self._config.cert, self._config.key)

        site = web.TCPSite(
            runner, self._config.host, self._config.port, ssl_context=ssl_context,
        )
        try:
            await site.start()
            scheme = "https" if ssl_context else "http"
            print(f"[teleop] serving on {scheme}://{self._config.host}:{self._config.port}")
            await self._shutdown.wait()
        finally:
            await runner.cleanup()
            await self._hand.stop()

    def _make_app(self) -> web.Application:
        static_dir = Path(self._config.static_dir)
        app = web.Application()
        app.router.add_get("/ws", self._handle_ws)
        # Serve index.html at "/" explicitly, then fall through to static
        # files for everything else under the static dir.
        app.router.add_get(
            "/",
            lambda _req: web.FileResponse(static_dir / "index.html"),
        )
        app.router.add_static("/", path=str(static_dir), show_index=False)
        return app

    async def _handle_ws(self, request) -> web.WebSocketResponse:
        """Main WebSocket handler. One connection -> one control loop."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        await ws.send_str(encode(PhaseMsg(phase=self._phase)))
        await ws.send_str(encode(PromptMsg(text=self._calib.current_prompt)))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        decoded = decode(msg.data)
                    except Exception as exc:
                        print(f"[teleop] control decode error: {exc!r}; raw={msg.data!r}")
                        continue
                    if isinstance(decoded, HandStateMsg):
                        self._latest_hand = decoded
                        if self._phase == "ready" and decoded.valid:
                            await self._send_curls(decoded)
                    elif isinstance(decoded, ButtonMsg):
                        await self._on_button(ws, decoded)
                elif msg.type == WSMsgType.ERROR:
                    print(f"[teleop] ws error: {ws.exception()!r}")
                    break
        finally:
            if not ws.closed:
                await ws.close()
        return ws

    async def _on_button(self, ws: web.WebSocketResponse, btn: ButtonMsg) -> None:
        """Handle a single rising-edge gesture event."""
        if btn.hand != "left" or not btn.pressed or btn.name != "x_click":
            return
        await self._advance_calibration(ws)

    async def _advance_calibration(self, ws: web.WebSocketResponse) -> None:
        """Drive the finger-calibration FSM from a left-hand pinch."""
        if self._phase == "idle":
            self._calib.on_start()
            self._phase = "finger_cal"
            await ws.send_str(encode(PhaseMsg(phase=self._phase)))
            await ws.send_str(encode(PromptMsg(text=self._calib.current_prompt)))
            return

        if self._phase == "finger_cal":
            if not self._latest_hand.valid:
                await ws.send_str(encode(PromptMsg(
                    text=self._calib.current_prompt
                         + "\n(no hand tracked -- bring your right hand into view)",
                    severity="warn",
                )))
                return
            self._calib.on_confirm(self._latest_hand.curls, self._latest_hand.abduction)
            if self._calib.is_complete:
                self._phase = "ready"
                rec = self._calib.record
                print(
                    f"[teleop] calibration complete: "
                    f"curl_min={rec.min_curl.tolist()} curl_max={rec.max_curl.tolist()} "
                    f"abd_min={rec.min_abd:.3f} abd_max={rec.max_abd:.3f}"
                )
                await ws.send_str(encode(PhaseMsg(phase=self._phase)))
                await ws.send_str(encode(PromptMsg(
                    text="Ready. Streaming finger tracking to the hand.",
                )))
            else:
                await ws.send_str(encode(PromptMsg(text=self._calib.current_prompt)))

    async def _send_curls(self, hand_msg: HandStateMsg) -> None:
        raw_curls = np.asarray(hand_msg.curls, dtype=np.float32)
        curls = self._calib.record.apply_curl(raw_curls)
        abd = self._calib.record.apply_abduction(float(hand_msg.abduction))
        try:
            await self._hand.send(curls, abd)
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_send_error_log >= _SEND_ERROR_LOG_PERIOD_S:
                self._last_send_error_log = now
                print(f"[teleop] hand send error: {exc!r}")
