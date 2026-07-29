// Pure: finger curl + thumb abduction from joint positions.
//
// Direct port of ../vr_tendon_arm_teleop/webxr_app/static/hand_math.js so
// the calibration numbers a user records on one project work on the other.
//
// Inputs are arrays of [x, y, z] (or null) per joint index, length 25.
// Returns:
//   fingerCurl(points, finger)   number in [0, 1]
//   allCurls(points)             [thumb, index, middle, ring, little]
//   thumbAbduction(points)       signed radians

export const JOINT_COUNT = 25;
export const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'little'];
export const FINGER_THUMB  = 0;
export const FINGER_INDEX  = 1;
export const FINGER_MIDDLE = 2;
export const FINGER_RING   = 3;
export const FINGER_LITTLE = 4;

// Per-finger joint chains (metacarpal -> tip), matching the iteration
// order of `for (const space of inputSource.hand.values())` in the Quest
// browser.
const FINGER_CHAINS = [
  [1, 2, 3, 4],
  [5, 6, 7, 8, 9],
  [10, 11, 12, 13, 14],
  [15, 16, 17, 18, 19],
  [20, 21, 22, 23, 24],
];

const JOINT_WRIST = 0;
const JOINT_INDEX_METACARPAL  = 5;
const JOINT_LITTLE_METACARPAL = 20;
const JOINT_THUMB_METACARPAL  = 1;
const JOINT_THUMB_PROXIMAL    = 2;
const JOINT_THUMB_TIP = 4;
const JOINT_INDEX_TIP = 9;

export function fingerCurl(points, finger) {
  const chain = FINGER_CHAINS[finger];
  const joints = [];
  for (const idx of chain) {
    const p = points[idx];
    if (!p) return 0.0;
    joints.push(p);
  }
  const bones = [];
  for (let i = 0; i < joints.length - 1; i++) {
    const a = joints[i], b = joints[i + 1];
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const len = Math.hypot(d[0], d[1], d[2]);
    if (len < 1e-6) return 0.0;
    bones.push([d[0] / len, d[1] / len, d[2] / len]);
  }
  let bend = 0.0;
  for (let i = 0; i < bones.length - 1; i++) {
    let c = bones[i][0] * bones[i + 1][0]
          + bones[i][1] * bones[i + 1][1]
          + bones[i][2] * bones[i + 1][2];
    c = Math.max(-1.0, Math.min(1.0, c));
    bend += Math.acos(c);
  }
  return Math.max(0.0, Math.min(1.0, bend / (2.0 * Math.PI)));
}

export function allCurls(points) {
  return [0, 1, 2, 3, 4].map(f => fingerCurl(points, f));
}

// Distance (m) between thumb-tip and index-tip -- used to detect a pinch
// gesture (thumb + index touching). Returns null when either joint is
// untracked.
export function pinchDistance(points) {
  const a = points[JOINT_THUMB_TIP];
  const b = points[JOINT_INDEX_TIP];
  if (!a || !b) return null;
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function thumbAbduction(points) {
  const wrist    = points[JOINT_WRIST];
  const idx_mc   = points[JOINT_INDEX_METACARPAL];
  const lit_mc   = points[JOINT_LITTLE_METACARPAL];
  const thumb_mc = points[JOINT_THUMB_METACARPAL];
  const thumb_pr = points[JOINT_THUMB_PROXIMAL];
  if (!wrist || !idx_mc || !lit_mc || !thumb_mc || !thumb_pr) return 0.0;

  const a = [idx_mc[0] - wrist[0], idx_mc[1] - wrist[1], idx_mc[2] - wrist[2]];
  const b = [lit_mc[0] - wrist[0], lit_mc[1] - wrist[1], lit_mc[2] - wrist[2]];
  const n = [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const nlen = Math.hypot(n[0], n[1], n[2]);
  const bone = [thumb_pr[0] - thumb_mc[0], thumb_pr[1] - thumb_mc[1], thumb_pr[2] - thumb_mc[2]];
  const blen = Math.hypot(bone[0], bone[1], bone[2]);
  if (nlen < 1e-6 || blen < 1e-6) return 0.0;
  let sin_a = (bone[0] * n[0] + bone[1] * n[1] + bone[2] * n[2]) / (nlen * blen);
  sin_a = Math.max(-1.0, Math.min(1.0, sin_a));
  return Math.asin(sin_a);
}
