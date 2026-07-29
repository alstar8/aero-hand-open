// Renders a rigged hand FBX at the WebXR wrist pose and animates fingers
// from per-frame WebXR joint POSITIONS (orientation conventions vary too
// much between rigs and the WebXR spec to copy quaternions directly).
//
// Per-bone algorithm:
//   1. At load time, for each finger bone, compute and cache:
//        restLocalQ      -- the FBX rest local quaternion
//        restParentDir   -- unit vector from this bone's origin to the
//                           next bone in the chain, expressed in the
//                           PARENT'S frame.
//   2. Per frame, for each bone, compute the desired direction (from
//      the WebXR joint at this bone to the next joint along the finger)
//      and re-express it in the parent's CURRENT world frame.
//   3. qAlign = unit-vector rotation from restParentDir to desiredParentDir.
//      bone.quaternion = qAlign * restLocalQ.
//      That keeps the rest pose's twist around the bone axis while bending
//      it along the finger.
//
// Asset note: bone-name regex accepts both b_l_* and b_r_* prefixes so
// the same code works for either hand FBX. Pass `url: './assets/LeftHand.fbx'`
// to swap.

import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

const DEFAULT_URL = './assets/RightHand.fbx';
const FALLBACK_COLOR = 0xeecbbf;

// WebXR joint index (per the Hand Input iteration order) -> Oculus FBX
// bone-name suffix. Metacarpals for thumb/index/middle/ring are fused
// into the palm in this rig (no bones); pinky's metacarpal is separate.
const JOINT_TO_BONE_SUFFIX = {
  2: 'thumb1',  3: 'thumb2',  4: 'thumb3',
  6: 'index1',  7: 'index2',  8: 'index3',
  11: 'middle1', 12: 'middle2', 13: 'middle3',
  16: 'ring1', 17: 'ring2', 18: 'ring3',
  20: 'pinky0', 21: 'pinky1', 22: 'pinky2', 23: 'pinky3',
};

// For each driven bone, the next WebXR joint index along the finger;
// used to compute the desired bone direction per frame.
const JOINT_NEXT = {
  2: 3,  3: 4,
  6: 7,  7: 8,  8: 9,
  11: 12, 12: 13, 13: 14,
  16: 17, 17: 18, 18: 19,
  20: 21, 21: 22, 22: 23, 23: 24,
};

// Map a driven joint index to the finger it belongs to (0=thumb..4=pinky).
// Used by the curl-based bend path (driveCurls) below.
const JOINT_TO_FINGER_IDX = {
  2: 0, 3: 0, 4: 0,
  6: 1, 7: 1, 8: 1,
  11: 2, 12: 2, 13: 2,
  16: 3, 17: 3, 18: 3,
  20: 4, 21: 4, 22: 4, 23: 4,
};

// Per-segment bend at full curl. Distal segments bend more than proximal
// in reality, but a single value reads fine for the alignment-hint use
// case; tune later if needed.
const SEGMENT_CURL_MAX_RAD = (Math.PI / 180) * 70;

const _tmpParentInv = new THREE.Matrix4();
const _tmpDir       = new THREE.Vector3();
const _tmpQ         = new THREE.Quaternion();

export class HandView {
  constructor(scene, {
    url = DEFAULT_URL,
    scale = 0.01,
    offsetPosition = [-0.03, -0.05, -0.1],
    // 180° around Y. The FBX rest has the hand laid out so its long axis
    // (fingers) and the wrist orientation from WebXR disagree by a half-turn
    // around the wrist's vertical (palm-normal) axis. 180°X (palm flip) made
    // it appear mirrored; identity left the wrist pointing away.
    offsetQuaternion = [0, 1, 0, 0],
    // When set, every mesh in the FBX is replaced with a flat-coloured
    // material (no texture). Used for the "ghost" hand that marks the
    // re-engage pose -- distinct colour + translucency makes it obvious.
    color = null,
    opacity = 1.0,
    debugJointDots = true,
  } = {}) {
    this._forceColor = color;
    this._forceOpacity = opacity;
    this._group = new THREE.Group();
    this._group.visible = false;
    scene.add(this._group);

    // DEBUG: 25 cyan spheres parented to the world scene, one per WebXR
    // joint, drawn at the raw positions reported by the headset. Useful
    // for sanity-checking that the rig orientation matches reality.
    // Skipped for the ghost hand so it doesn't double up with the live one.
    this._debugDots = [];
    if (debugJointDots) {
      const dotGeo = new THREE.SphereGeometry(0.006, 8, 8);
      const dotMat = new THREE.MeshBasicMaterial({ color: 0x00ffff });
      for (let i = 0; i < 25; i++) {
        const m = new THREE.Mesh(dotGeo, dotMat);
        m.visible = false;
        scene.add(m);
        this._debugDots.push(m);
      }
      // Highlight the wrist (joint 0) so we can see which end is which.
      this._debugDots[0].material = new THREE.MeshBasicMaterial({ color: 0xff00ff });
    }

    this._inner = new THREE.Group();
    this._inner.position.set(...offsetPosition);
    this._inner.quaternion.set(...offsetQuaternion);
    this._inner.scale.setScalar(scale);
    this._group.add(this._inner);

    // jointIdx -> { bone, restLocalQ, restParentDir }
    this._bones = {};

    new FBXLoader().load(url, (root) => {
      root.traverse((c) => {
        if (!c.isMesh) return;
        if (this._forceColor != null) {
          // Ghost-hand path: force a flat coloured (and optionally
          // translucent) material regardless of what the FBX shipped with.
          c.material = new THREE.MeshStandardMaterial({
            color: this._forceColor,
            roughness: 0.6,
            metalness: 0.0,
            transparent: this._forceOpacity < 1.0,
            opacity: this._forceOpacity,
            depthWrite: this._forceOpacity >= 1.0,
          });
          return;
        }
        const m = Array.isArray(c.material) ? c.material[0] : c.material;
        if (!m || m.map == null) {
          c.material = new THREE.MeshStandardMaterial({
            color: FALLBACK_COLOR, roughness: 0.6, metalness: 0.0,
          });
        }
      });

      this._inner.add(root);
      // Stamp world matrices so child.position can be interpreted later.
      this._inner.updateMatrixWorld(true);

      // Build a name -> bone index lookup for the suffixes we care about.
      const suffixToJoint = {};
      for (const [j, s] of Object.entries(JOINT_TO_BONE_SUFFIX)) {
        suffixToJoint[s] = parseInt(j);
      }

      const allBoneNames = [];
      let wristBone = null;
      // First pass: collect bones by joint index.
      root.traverse((c) => {
        if (!c.isBone) return;
        allBoneNames.push(c.name);
        if (!wristBone && /wrist/i.test(c.name)) wristBone = c;
        const m = c.name.match(/b_[lr]_([a-z]+\d)/);
        if (!m) return;
        const j = suffixToJoint[m[1]];
        if (j === undefined) return;
        this._bones[j] = { bone: c, restLocalQ: c.quaternion.clone(), restParentDir: null };
      });

      // Anchor the FBX wrist bone exactly on the WebXR wrist joint. The
      // user-supplied offsetPosition becomes an additional nudge on top
      // of this auto-alignment. Without this, the model's wrist sits at
      // wherever the FBX root happens to be, not at the tracked wrist.
      if (wristBone) {
        wristBone.updateMatrixWorld(true);
        const wristWorld = new THREE.Vector3().setFromMatrixPosition(wristBone.matrixWorld);
        const wristInGroup = this._group.worldToLocal(wristWorld.clone());
        this._inner.position.sub(wristInGroup);
        this._inner.updateMatrixWorld(true);
      }

      // Second pass: for each driven bone, find the next bone along the
      // chain (preferring the WebXR-mapped child if present, falling back
      // to the first bone child for tip bones whose "next joint" has no
      // FBX bone). Compute restParentDir = direction from this bone to
      // that child, expressed in this bone's PARENT frame.
      for (const j in this._bones) {
        const { bone, restLocalQ } = this._bones[j];
        const nextJ = JOINT_NEXT[j];
        let nextBone = null;
        if (nextJ != null && this._bones[nextJ]) nextBone = this._bones[nextJ].bone;
        if (!nextBone) {
          nextBone = bone.children.find(ch => ch.isBone) || null;
        }
        if (!nextBone) continue;
        // nextBone.position is in `bone`'s local frame. Direction from
        // bone origin to nextBone origin in PARENT'S frame = restLocalQ * nextBone.position.
        const dir = nextBone.position.clone().applyQuaternion(restLocalQ);
        if (dir.lengthSq() < 1e-12) continue;
        dir.normalize();
        this._bones[j].restParentDir = dir;
      }

      const mapped = Object.values(this._bones).filter(b => b.restParentDir).length;
      const box = new THREE.Box3().setFromObject(root);
      console.log('[hand_view] loaded', url,
        'size', box.getSize(new THREE.Vector3()).toArray(),
        'bones', allBoneNames.length,
        'driveable', mapped);
    }, undefined, (err) => console.warn('[hand_view] load failed', err));
  }

  update(sample) {
    if (!sample || !sample.valid || !sample.wrist) {
      this._group.visible = false;
      for (const d of this._debugDots) d.visible = false;
      return;
    }
    this._group.visible = true;
    // DEBUG: place a dot at every reported joint position (world frame).
    const pts = sample.points || [];
    for (let i = 0; i < this._debugDots.length; i++) {
      const p = pts[i];
      const dot = this._debugDots[i];
      if (p) {
        dot.position.set(p[0], p[1], p[2]);
        dot.visible = true;
      } else {
        dot.visible = false;
      }
    }
    this._group.position.set(sample.wrist[0], sample.wrist[1], sample.wrist[2]);
    if (sample.wristOrientation) {
      const q = sample.wristOrientation;
      this._group.quaternion.set(q[0], q[1], q[2], q[3]);
    }
    if (sample.points) {
      // Make sure parent world matrices reflect the new wrist transform
      // before we start computing bone-local directions off them.
      this._group.updateMatrixWorld(true);
      this._driveBones(sample.points);
    }
  }

  _driveBones(points) {
    // Process bones in joint-index order: index order corresponds to
    // depth along each finger (proximal -> tip), so each bone's parent
    // matrixWorld is up-to-date by the time we process it.
    const indices = Object.keys(this._bones).map(Number).sort((a, b) => a - b);
    for (const j of indices) {
      const info = this._bones[j];
      if (!info.restParentDir) continue;
      const nextJ = JOINT_NEXT[j];
      if (nextJ == null) continue;
      const p = points[j];
      const pNext = points[nextJ];
      if (!p || !pNext) continue;

      _tmpDir.set(pNext[0] - p[0], pNext[1] - p[1], pNext[2] - p[2]);
      if (_tmpDir.lengthSq() < 1e-10) continue;
      _tmpDir.normalize();

      // Re-express the desired world direction in the bone's parent local frame.
      _tmpParentInv.copy(info.bone.parent.matrixWorld).invert();
      _tmpDir.transformDirection(_tmpParentInv);

      _tmpQ.setFromUnitVectors(info.restParentDir, _tmpDir);
      info.bone.quaternion.copy(_tmpQ).multiply(info.restLocalQ);
      info.bone.updateMatrix();
      info.bone.updateMatrixWorld(true);
    }
  }

  setOffset(position, quaternion) {
    if (position) this._inner.position.set(...position);
    if (quaternion) this._inner.quaternion.set(...quaternion);
  }

  setVisible(v) {
    this._group.visible = !!v;
    if (!v) for (const d of this._debugDots) d.visible = false;
  }

  // Bend finger bones from a 5-vector of normalized curl values
  // (thumb..pinky, 0..1). Used by the ghost hand to mirror the robot's
  // current finger state when we don't have full per-joint positions.
  //
  // For each bone we rotate the rest direction around a heuristic bend
  // axis (perpendicular to the bone's long axis, biased toward the palm
  // normal in the parent frame) and apply the resulting alignment quat
  // on top of the bone's rest local rotation -- same shape as the
  // position-driven path in _driveBones.
  driveCurls(curls) {
    if (!curls) return;
    const indices = Object.keys(this._bones).map(Number).sort((a, b) => a - b);
    const _palmGuess = new THREE.Vector3(0, 1, 0);   // parent-local heuristic
    const _bendAxis = new THREE.Vector3();
    const _desired = new THREE.Vector3();
    const _qBend = new THREE.Quaternion();
    const _qAlign = new THREE.Quaternion();
    for (const j of indices) {
      const info = this._bones[j];
      if (!info.restParentDir) continue;
      const fingerIdx = JOINT_TO_FINGER_IDX[j];
      if (fingerIdx == null) continue;
      const c = Math.max(0, Math.min(1, curls[fingerIdx] ?? 0));
      const angle = c * SEGMENT_CURL_MAX_RAD;
      _bendAxis.crossVectors(info.restParentDir, _palmGuess);
      if (_bendAxis.lengthSq() < 1e-8) _bendAxis.set(0, 0, 1);
      _bendAxis.normalize();
      _qBend.setFromAxisAngle(_bendAxis, angle);
      _desired.copy(info.restParentDir).applyQuaternion(_qBend);
      _qAlign.setFromUnitVectors(info.restParentDir, _desired);
      info.bone.quaternion.copy(_qAlign).multiply(info.restLocalQ);
      info.bone.updateMatrix();
      info.bone.updateMatrixWorld(true);
    }
  }
}
