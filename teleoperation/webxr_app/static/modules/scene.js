// Three.js scene setup: renderer with renderer.xr.enabled, ambient/dir
// lights, the camera-as-rig that the overlay and HUD attach to (so
// they're head-locked). Runs as an `immersive-ar` session on Quest so
// passthrough shows the real room behind our content (point cloud,
// hands, controller, overlay) — like the Oculus system menu.
//
// Surface:
//   const s = new Scene(domElement);
//   s.add(mesh);             // anything that should render in the world frame
//   s.addHeadLocked(mesh);   // attached to the XR camera; moves with the head
//   s.setAnimationLoop(fn);
//   await s.startSession();

import * as THREE from 'three';

export class Scene {
  constructor(canvasParent = document.body) {
    this.scene = new THREE.Scene();
    this.scene.background = null;

    this.camera = new THREE.PerspectiveCamera(
      70, window.innerWidth / window.innerHeight, 0.05, 100,
    );
    // Eye height for the desktop preview; in VR the headset overrides this.
    this.camera.position.set(0, 1.6, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.xr.enabled = true;
    this.renderer.xr.setReferenceSpaceType('local-floor');
    this.renderer.setAnimationLoop((time, xrFrame) => this._tick(time, xrFrame));
    canvasParent.appendChild(this.renderer.domElement);

    // Head-locked group: child of the camera, so it follows head motion.
    this._headLocked = new THREE.Group();
    this.camera.add(this._headLocked);
    this.scene.add(this.camera);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(1, 2, 1);
    this.scene.add(dir);

    this._animFn = null;
    window.addEventListener('resize', () => this._onResize());
  }

  add(obj) { this.scene.add(obj); }
  addHeadLocked(obj) { this._headLocked.add(obj); }

  setAnimationLoop(fn) { this._animFn = fn; }

  async startSession() {
    if (this.renderer.xr.isPresenting) return;
    if (!navigator.xr) {
      throw new Error('WebXR not available in this browser');
    }

    // Try AR (passthrough) first, then VR. Within each mode, fall back from
    // richer feature sets to minimal ones — some Quest browser builds reject
    // unknown optional feature strings (or local-floor) with
    // "session configuration is not supported".
    const modes = [];
    if (await navigator.xr.isSessionSupported?.('immersive-ar')) {
      modes.push('immersive-ar');
    }
    if (await navigator.xr.isSessionSupported?.('immersive-vr')) {
      modes.push('immersive-vr');
    }
    if (!modes.length) {
      throw new Error(
        'Neither immersive-ar nor immersive-vr is supported. ' +
        'Open this page in the Meta Quest Browser over HTTPS.',
      );
    }

    const featureSets = [
      {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hand-tracking'],
        refSpace: 'local-floor',
      },
      {
        requiredFeatures: ['local-floor'],
        optionalFeatures: [
          'hand-tracking',
          'simultaneous-hands-and-controllers',
          'hand-input-with-controllers',
        ],
        refSpace: 'local-floor',
      },
      {
        // No required features — 'local' is always available.
        requiredFeatures: [],
        optionalFeatures: ['hand-tracking', 'local-floor'],
        refSpace: 'local',
      },
      {
        requiredFeatures: [],
        optionalFeatures: ['hand-tracking'],
        refSpace: 'local',
      },
    ];

    let session = null;
    let lastError = null;
    let usedRefSpace = 'local-floor';
    for (const mode of modes) {
      for (const features of featureSets) {
        try {
          session = await navigator.xr.requestSession(mode, {
            requiredFeatures: features.requiredFeatures,
            optionalFeatures: features.optionalFeatures,
          });
          usedRefSpace = features.refSpace;
          console.log(
            `[xr] started ${mode} ref=${usedRefSpace} ` +
            `required=${JSON.stringify(features.requiredFeatures)} ` +
            `optional=${JSON.stringify(features.optionalFeatures)}`,
          );
          break;
        } catch (e) {
          lastError = e;
          console.warn(
            `[xr] ${mode} failed ` +
            `(required=${JSON.stringify(features.requiredFeatures)}):`,
            e,
          );
        }
      }
      if (session) break;
    }

    if (!session) {
      throw lastError || new Error('No supported WebXR session configuration');
    }

    this.renderer.xr.setReferenceSpaceType(usedRefSpace);
    await this.renderer.xr.setSession(session);
  }

  _tick(time, xrFrame) {
    if (this._animFn) this._animFn(time, xrFrame);
    this.renderer.render(this.scene, this.camera);
  }

  _onResize() {
    if (this.renderer.xr.isPresenting) return;
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }
}
