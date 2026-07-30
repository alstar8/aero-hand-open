// WebSocket client.
//
// Two channels multiplexed onto one connection:
//   - JSON text frames: control messages (hand state, buttons, prompts, ...).
//   - Binary frames: point cloud payloads (see teleop_core/point_cloud.py
//     for the layout).
//
// Auto-reconnects with exponential backoff. Outbound JSON sent before the
// socket is open is queued and flushed on connect.
//
// Surface:
//   const comms = new Comms('/ws');
//   comms.onJson = (msg) => { ... };
//   comms.onBinary = (arrayBuffer) => { ... };
//   comms.sendJson({ type: 'hand', ... });

export class Comms {
  constructor(path = '/ws') {
    const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
    this._url = scheme + location.host + path;
    this._ws = null;
    this._outbox = [];
    this._reconnectDelay = 500;
    this._reconnectMax = 5000;
    this._closed = false;
    this.onJson = null;
    this.onBinary = null;
    this._open();
  }

  _open() {
    if (this._closed) return;
    const ws = new WebSocket(this._url);
    ws.binaryType = 'arraybuffer';

    ws.addEventListener('open', () => {
      this._reconnectDelay = 500;
      for (const m of this._outbox) ws.send(m);
      this._outbox.length = 0;
    });

    ws.addEventListener('message', (ev) => {
      if (typeof ev.data === 'string') {
        if (!this.onJson) return;
        try { this.onJson(JSON.parse(ev.data)); }
        catch (e) { console.error('comms: bad JSON', e, ev.data); }
      } else if (this.onBinary) {
        this.onBinary(ev.data);
      }
    });

    ws.addEventListener('close', () => {
      this._ws = null;
      if (this._closed) return;
      setTimeout(() => this._open(), this._reconnectDelay);
      this._reconnectDelay = Math.min(this._reconnectMax, this._reconnectDelay * 2);
    });

    ws.addEventListener('error', (e) => {
      console.warn('comms: ws error', e);
    });

    this._ws = ws;
  }

  sendJson(obj) {
    const s = JSON.stringify(obj);
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(s);
    } else {
      this._outbox.push(s);
    }
  }

  close() {
    this._closed = true;
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
  }
}
