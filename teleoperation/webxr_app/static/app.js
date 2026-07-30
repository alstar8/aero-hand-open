// Entry point. Wires together the JS modules -- deliberately small: this
// teleop only ever streams right-hand finger curls + thumb abduction to
// the server, gated by a left-hand pinch calibration flow. No wrist
// tracking, no workspace, no point cloud, no controllers.

import { Scene } from './modules/scene.js';
import { Comms } from './modules/comms.js';
import { Overlay } from './modules/overlay.js';
import { StateMachine } from './modules/state_machine.js';
import { InputReader } from './modules/input_reader.js';
import { HandView } from './modules/hand_view.js';
import { allCurls, thumbAbduction, pinchDistance } from './modules/hand_math.js';

const scene = new Scene(document.body);
const overlay = new Overlay(scene);
const stateMachine = new StateMachine(overlay);
const input = new InputReader();
const rightHandView = new HandView(scene);   // right bare hand, visual feedback only
const comms = new Comms('/ws');

comms.onJson = (msg) => {
  switch (msg.type) {
    case 'phase':  stateMachine.setPhase(msg.phase); break;
    case 'prompt': stateMachine.applyPrompt(msg); break;
    default:       console.log('[control]', msg);
  }
};

const HAND_SEND_PERIOD_MS = 1000 / 30;
let lastHandSendMs = 0;

// Left-hand pinch (thumb tip + index tip touching) advances calibration.
// Bare-hand gesture on purpose -- no controller needed at all, so both
// hands stay tracked simultaneously through calibration. Hysteresis
// avoids chatter right at the threshold.
const PINCH_ENGAGE_M  = 0.025;
const PINCH_RELEASE_M = 0.04;
let leftPinching = false;

scene.setAnimationLoop((time, frame) => {
  if (!frame) return;
  const ref = scene.renderer.xr.getReferenceSpace();
  if (!ref) return;

  const snap = input.read(frame, ref);

  const right = snap.hands.right;
  rightHandView.update(right);
  if (right && (time - lastHandSendMs) >= HAND_SEND_PERIOD_MS) {
    lastHandSendMs = time;
    comms.sendJson({
      type: 'hand',
      curls: allCurls(right.points),
      abduction: thumbAbduction(right.points),
      valid: right.valid,
    });
  }

  const leftHand = snap.hands.left;
  const pinchDist = leftHand && leftHand.valid ? pinchDistance(leftHand.points) : null;
  if (pinchDist != null) {
    if (!leftPinching && pinchDist <= PINCH_ENGAGE_M) {
      leftPinching = true;
      comms.sendJson({ type: 'button', hand: 'left', name: 'x_click', pressed: true });
    } else if (leftPinching && pinchDist >= PINCH_RELEASE_M) {
      leftPinching = false;
    }
  }
});

const enterBtn = document.getElementById('enter');
if (enterBtn) {
  if (!navigator.xr) {
    enterBtn.disabled = true;
    enterBtn.textContent = 'WebXR unavailable (open on Quest browser)';
    const info = document.getElementById('info');
    if (info) {
      info.insertAdjacentHTML(
        'beforeend',
        '<br><br><b>This page must be opened in the Meta Quest browser</b> ' +
        '(HTTPS to your PC LAN IP). Desktop Chrome/Firefox cannot start ' +
        'immersive VR/AR here — there is no video stream to watch on a PC.',
      );
    }
  } else {
    enterBtn.addEventListener('click', async () => {
      try {
        enterBtn.disabled = true;
        enterBtn.textContent = 'Starting XR…';
        await scene.startSession();
        enterBtn.textContent = 'In XR';
      } catch (e) {
        console.warn('XR session start failed:', e);
        enterBtn.disabled = false;
        const reason = (e && (e.message || String(e))) || 'unknown error';
        enterBtn.textContent = 'XR start failed';
        enterBtn.title = reason;
        const info = document.getElementById('info');
        if (info) {
          info.insertAdjacentHTML(
            'beforeend',
            `<br><br><span style="color:#c44"><b>XR start failed:</b> ${reason}</span>`,
          );
        }
      }
    });
  }
}
