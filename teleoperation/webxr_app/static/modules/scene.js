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
    // Prefer immersive-ar so Quest enables passthrough: the operator
    // sees their real room and our content (point cloud, hands,
    // controller, overlay) renders on top. Fall back to immersive-vr
    // if AR isn't supported.
    const optionalFeatures = [
      'hand-tracking',
      'simultaneous-hands-and-controllers',
      'hand-input-with-controllers',
    ];
    let session;
    const arSupported = await navigator.xr.isSessionSupported?.('immersive-ar');
    if (arSupported) {
      session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor'],
        optionalFeatures,
      });
    } else {
      session = await navigator.xr.requestSession('immersive-vr', {
        requiredFeatures: ['local-floor'],
        optionalFeatures,
      });
    }
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
