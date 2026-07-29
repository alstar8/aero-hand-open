// Head-locked text panel. Two channels:
//   - prompt:  large primary text (calibration prompt, "Pull trigger…")
//   - warning: smaller row below, color-coded by severity
//
// Both are CanvasTexture-backed planes attached to the head rig via
// Scene.addHeadLocked, so they follow the user's gaze.

import * as THREE from 'three';

const SEVERITY_COLORS = {
  info:  '#ffffff',
  warn:  '#ffcc44',
  error: '#ff5544',
  ok:    '#88ff88',
};

function makePanel(scene, position, size, canvasW, canvasH) {
  const canvas = document.createElement('canvas');
  canvas.width = canvasW; canvas.height = canvasH;
  const ctx = canvas.getContext('2d');
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({
    map: tex, transparent: true, depthTest: false,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size[0], size[1]), mat);
  mesh.position.set(...position);
  mesh.renderOrder = 999;
  mesh.visible = false;
  scene.addHeadLocked(mesh);
  return { canvas, ctx, tex, mesh };
}

function wrapLine(ctx, text, maxWidth) {
  const words = text.split(/\s+/);
  const out = [];
  let cur = '';
  for (const w of words) {
    const candidate = cur ? cur + ' ' + w : w;
    if (ctx.measureText(candidate).width <= maxWidth || !cur) {
      cur = candidate;
    } else {
      out.push(cur);
      cur = w;
    }
  }
  if (cur) out.push(cur);
  return out;
}

function drawText(panel, text, color, fontPx) {
  const { canvas, ctx, tex, mesh } = panel;
  if (text == null || text === '') {
    mesh.visible = false;
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.78)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = color;
  ctx.font = `bold ${fontPx}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const padding = 40;
  const maxWidth = canvas.width - 2 * padding;
  const lines = [];
  for (const raw of String(text).split('\n')) {
    for (const wrapped of wrapLine(ctx, raw, maxWidth)) lines.push(wrapped);
  }
  const lineHeight = fontPx + 10;
  const total = lineHeight * lines.length;
  let y = (canvas.height - total) / 2 + lineHeight / 2;
  for (const line of lines) {
    ctx.fillText(line, canvas.width / 2, y);
    y += lineHeight;
  }
  tex.needsUpdate = true;
  mesh.visible = true;
}

export class Overlay {
  constructor(scene) {
    this._prompt = makePanel(scene, [0, 0.05, -1.0], [1.4, 0.34], 1536, 384);
    this._warning = makePanel(scene, [0, -0.32, -0.95], [1.2, 0.1], 1280, 104);
  }

  setPrompt(text, severity = 'info') {
    drawText(this._prompt, text, SEVERITY_COLORS[severity] || '#ffffff', 44);
  }

  setWarning(text, severity = 'warn') {
    drawText(this._warning, text, SEVERITY_COLORS[severity] || '#ffcc44', 28);
  }

  clear() {
    this._prompt.mesh.visible = false;
    this._warning.mesh.visible = false;
  }
}
