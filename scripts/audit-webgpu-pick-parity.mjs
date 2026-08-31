#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';

const targets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const page = targets.find(target =>
  target.type === 'page' && target.url.includes(`:${pagePort}/`),
);
assert.ok(page, `no Hyperscope page found on port ${pagePort}`);

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});

function auditPickParity() {
  const app = globalThis.__hyperscopeAppShadowController;
  if (!app) throw new Error('Rust AppShadow is unavailable');
  const canvas = document.getElementById('cv');
  if (!(canvas instanceof HTMLCanvasElement) || canvas.width <= 0 || canvas.height <= 0) {
    throw new Error('Hyperscope canvas is unavailable');
  }

  // This independent oracle reconstructs the URL-addressed manual camera. It
  // deliberately declines authored-camera and walking routes rather than
  // borrowing private browser controller state.
  const route = new URLSearchParams(location.search);
  if (route.has('walk') || route.has('camera')) {
    throw new Error('live pick audit currently requires the manual URL camera');
  }
  const number = (name, fallback) => Number(route.get(name) ?? fallback);
  const rotX = number('rx', 0);
  const rotY = number('ry', 0);
  const roll = number('rz', 0);
  const pan = [number('px', 0), number('py', 0), number('pz', 0)];
  const zoom = number('zoom', 4);
  const fov = number('fov', 45) * Math.PI / 180;
  const cosY = Math.cos(rotY);
  const sinY = Math.sin(rotY);
  const cosX = Math.cos(rotX);
  const sinX = Math.sin(rotX);
  const cosR = Math.cos(roll);
  const sinR = Math.sin(roll);
  const baseRight = [cosY, 0, -sinY];
  const baseUp = [-sinY * sinX, cosX, -cosY * sinX];
  const basis = [
    baseRight[0] * cosR + baseUp[0] * sinR,
    baseRight[1] * cosR + baseUp[1] * sinR,
    baseRight[2] * cosR + baseUp[2] * sinR,
    baseUp[0] * cosR - baseRight[0] * sinR,
    baseUp[1] * cosR - baseRight[1] * sinR,
    baseUp[2] * cosR - baseRight[2] * sinR,
    -sinY * cosX,
    -sinX,
    -cosY * cosX,
  ];
  const eye = pan.map((value, axis) => value - basis[6 + axis] * zoom);
  const up = basis.slice(3, 6);

  const view = new Float32Array(16);
  let forward = pan.map((value, axis) => value - eye[axis]);
  let length = Math.hypot(...forward);
  forward = forward.map(value => value / length);
  let side = [
    forward[1] * up[2] - forward[2] * up[1],
    forward[2] * up[0] - forward[0] * up[2],
    forward[0] * up[1] - forward[1] * up[0],
  ];
  length = Math.hypot(...side);
  side = side.map(value => value / length);
  const cameraUp = [
    side[1] * forward[2] - side[2] * forward[1],
    side[2] * forward[0] - side[0] * forward[2],
    side[0] * forward[1] - side[1] * forward[0],
  ];
  view.set([
    side[0], cameraUp[0], -forward[0], 0,
    side[1], cameraUp[1], -forward[1], 0,
    side[2], cameraUp[2], -forward[2], 0,
    -side.reduce((sum, value, axis) => sum + value * eye[axis], 0),
    -cameraUp.reduce((sum, value, axis) => sum + value * eye[axis], 0),
    forward.reduce((sum, value, axis) => sum + value * eye[axis], 0),
    1,
  ]);

  const projection = new Float32Array(16);
  const near = 0.01;
  const far = 10000;
  const cotangent = 1 / Math.tan(fov * 0.5);
  const depth = 1 / (near - far);
  projection[0] = cotangent / (canvas.width / canvas.height);
  projection[5] = cotangent;
  projection[10] = (near + far) * depth;
  projection[11] = -1;
  projection[14] = near * far * depth * 2;
  const mvp = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      mvp[column * 4 + row] = projection[row] * view[column * 4]
        + projection[4 + row] * view[column * 4 + 1]
        + projection[8 + row] * view[column * 4 + 2]
        + projection[12 + row] * view[column * 4 + 3];
    }
  }

  const center = [Math.floor(canvas.width * 0.5), Math.floor(canvas.height * 0.5)];
  const left = Math.floor(canvas.width * 0.4);
  const upper = Math.floor(canvas.height * 0.4);
  const samples = [
    center,
    [center[0], canvas.height - 1 - center[1]],
    [left, center[1]],
    [canvas.width - 1 - left, center[1]],
    [center[0], upper],
    [center[0], canvas.height - 1 - upper],
  ];

  return (async () => {
    const graphics = globalThis.__hyperscopeGraphicsBackend;
    if (!graphics) throw new Error('graphics diagnostics are unavailable');
    const wasPlaying = Boolean(app.snapshot()?.animationPlaying);
    const togglePlayback = () => window.dispatchEvent(new KeyboardEvent('keydown', {
      code: 'Space',
      key: ' ',
    }));
    if (wasPlaying) {
      const beforePause = await graphics.refresh();
      togglePlayback();
      let pausedFrameReady = false;
      for (let attempt = 0; attempt < 50; attempt += 1) {
        const afterPause = await graphics.refresh();
        if (afterPause?.lastFrameRevision > beforePause.lastFrameRevision
            && afterPause?.pickFrameReady) {
          pausedFrameReady = true;
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      if (!pausedFrameReady) {
        togglePlayback();
        throw new Error('WebGPU did not publish the final paused animation frame');
      }
    }
    try {
      let residency = null;
      for (let attempt = 0; attempt < 50; attempt += 1) {
        residency = await graphics.refresh();
        if (residency?.pickFrameReady) break;
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      if (!residency?.pickFrameReady) {
        throw new Error('WebGPU did not publish a coherent pick frame');
      }
      const reports = [];
      for (const [x, y] of samples) {
        const staged = app.stageBackendPickEvidence(
          mvp,
          view,
          new Float32Array(eye),
          x,
          y,
        );
        if (!staged?.staged) {
          throw new Error(`pick ${x},${y} rejected: ${staged?.error || 'unknown error'}`);
        }
        const diagnostics = await app.readBackendPickEvidence();
        reports.push(diagnostics.lastReport);
      }
      const finalResidency = await graphics.refresh();
      return {
        viewport: [canvas.width, canvas.height],
        reports,
        diagnostics: app.backendPickDiagnostics(),
        graphics: {
          requested: graphics.requested,
          effective: graphics.effective,
          state: graphics.state,
          presentationArmed: graphics.presentationArmed,
          residency: finalResidency,
        },
      };
    } finally {
      if (wasPlaying) togglePlayback();
    }
  })();
}

const response = await new Promise((resolve, reject) => {
  const id = 1;
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id !== id) return;
    if (message.error) reject(new Error(JSON.stringify(message.error)));
    else resolve(message.result);
  });
  socket.send(JSON.stringify({
    id,
    method: 'Runtime.evaluate',
    params: {
      expression: `(${auditPickParity.toString()})()`,
      awaitPromise: true,
      returnByValue: true,
    },
  }));
});
socket.close();

if (response.exceptionDetails) {
  throw new Error(response.exceptionDetails.exception?.description
    || response.exceptionDetails.text);
}
const audit = response.result.value;
console.log(JSON.stringify(audit, null, 2));
for (const report of audit.reports) {
  assert.equal(report.comparison.coverageMatches, true);
  assert.equal(report.comparison.identityMatches, true);
}
assert.equal(audit.graphics.requested, 'webgpu');
assert.equal(audit.graphics.effective, 'webgpu');
assert.equal(audit.graphics.residency.state, 'ready');
assert.equal(audit.graphics.residency.pickFrameReady, true);
assert.equal(audit.graphics.residency.frameFailures, 0);
