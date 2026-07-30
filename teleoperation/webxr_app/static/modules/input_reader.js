// Per-frame WebXR input snapshot.
//
// Walks session.inputSources once and produces a plain JS struct the
// rest of the app can consume without touching the WebXR API directly:
//
//   {
//     time:  DOMHighResTimeStamp,
//     hands: { left: HandSample | null, right: HandSample | null },
//     ctrls: { left: ControllerSample | null, right: ControllerSample | null },
//   }
//
// HandSample = {
//   points:        25 x [x,y,z] (or null),  // joint positions in reference space
//   orientations:  25 x [x,y,z,w] (or null), // joint orientations in reference space
//   valid:         bool,
//   wrist:         [x,y,z] | null,
//   wristOrientation: [x,y,z,w] | null,
// }
// ControllerSample = {
//   position: [x,y,z], orientation: [x,y,z,w],
//   buttons: { x, y, a, b, trigger, grip },   // bool except trigger/grip (analog 0..1)
// }
//
// Edge detection (X-click etc.) is done by the caller using the previous
// snapshot; the reader is stateless.

import { JOINT_COUNT } from './hand_math.js';

// Meta Quest Touch button layout, both controllers:
//   index 0: trigger (analog)
//   index 1: grip   (analog)
//   index 2: thumbstick press
//   index 3: thumbstick (axes)
//   index 4: primary (X on left, A on right)
//   index 5: secondary (Y on left, B on right)
const BTN_TRIGGER = 0;
const BTN_GRIP    = 1;
const BTN_PRIMARY = 4;
const BTN_SECOND  = 5;

function gpValue(gp, i) {
  return gp && gp.buttons && gp.buttons[i] ? gp.buttons[i].value : 0.0;
}
function gpPressed(gp, i) {
  return !!(gp && gp.buttons && gp.buttons[i] && gp.buttons[i].pressed);
}

export class InputReader {
  read(frame, referenceSpace) {
    const out = {
      time: performance.now(),
      hands: { left: null, right: null },
      ctrls: { left: null, right: null },
    };
    if (!frame || !referenceSpace) return out;

    const session = frame.session;
    for (const src of session.inputSources) {
      const handed = src.handedness;
      if (handed !== 'left' && handed !== 'right') continue;

      // --- Hand joints ---
      if (src.hand && frame.getJointPose) {
        const points = new Array(JOINT_COUNT).fill(null);
        const orientations = new Array(JOINT_COUNT).fill(null);
        let i = 0;
        let wrist = null, wristOri = null;
        for (const jointSpace of src.hand.values()) {
          if (i >= JOINT_COUNT) break;
          const jp = frame.getJointPose(jointSpace, referenceSpace);
          if (jp && jp.transform) {
            const p = jp.transform.position;
            const q = jp.transform.orientation;
            points[i] = [p.x, p.y, p.z];
            orientations[i] = [q.x, q.y, q.z, q.w];
            if (i === 0) {
              wrist = points[i];
              wristOri = orientations[i];
            }
          }
          i++;
        }
        const valid = !!(points[0] && points[5] && points[9]);
        out.hands[handed] = {
          points, orientations, valid, wrist, wristOrientation: wristOri,
        };
      }

      // --- Controller pose + buttons ---
      if (src.gamepad) {
        const space = src.gripSpace || src.targetRaySpace;
        const pose = space ? frame.getPose(space, referenceSpace) : null;
        const sample = {
          position: null,
          orientation: null,
          buttons: {
            trigger: gpValue(src.gamepad, BTN_TRIGGER),
            grip:    gpValue(src.gamepad, BTN_GRIP),
            primary: gpPressed(src.gamepad, BTN_PRIMARY),
            secondary: gpPressed(src.gamepad, BTN_SECOND),
          },
        };
        if (pose) {
          const p = pose.transform.position;
          const q = pose.transform.orientation;
          sample.position = [p.x, p.y, p.z];
          sample.orientation = [q.x, q.y, q.z, q.w];
        }
        out.ctrls[handed] = sample;
      }
    }
    return out;
  }
}
