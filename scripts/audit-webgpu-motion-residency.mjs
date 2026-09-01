#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const targets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const originalPage = targets.find(target =>
  target.type === 'page' && target.url.includes(`:${pagePort}/`));
assert.ok(originalPage, 'a pre-existing Hyperscope page is required');

const route = new URL(process.env.HYPERSCOPE_WEBGPU_MOTION_URL
  || `http://127.0.0.1:${pagePort}/?glb=horse.glb`);
for (const [key, value] of [
  ['gfx', 'webgpu'],
  ['mode', 'wire'],
  ['xform', 'sphere_reflection'],
  ['mr', '11.5'],
  ['minpx', '16'],
  ['animate', '1'],
  ['anim', '0'],
  ['fuzzy', '0'],
  ['lodimpl', 'rust'],
]) route.searchParams.set(key, value);
route.searchParams.set('_motion_residency_evidence', String(Date.now()));

const page = await (await fetch(
  `${cdpEndpoint}/json/new?${encodeURIComponent('about:blank')}`,
  { method: 'PUT' },
)).json();

async function activate(targetId) {
  const response = await fetch(`${cdpEndpoint}/json/activate/${targetId}`);
  if (!response.ok) throw new Error(`could not activate Chrome target ${targetId}`);
}

await activate(page.id);
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});

let nextCommandId = 1;
const pending = new Map();
const runtimeErrors = [];
socket.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id != null) {
    const waiter = pending.get(message.id);
    if (!waiter) return;
    pending.delete(message.id);
    if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
    else waiter.resolve(message.result);
    return;
  }
  if (message.method === 'Runtime.exceptionThrown') {
    runtimeErrors.push(message.params.exceptionDetails?.exception?.description
      || message.params.exceptionDetails?.text || 'runtime exception');
  }
  if (message.method === 'Runtime.consoleAPICalled'
      && message.params.type === 'error') {
    runtimeErrors.push(message.params.args.map(argument =>
      argument.value ?? argument.description ?? '').join(' '));
  }
  if (message.method === 'Log.entryAdded' && message.params.entry?.level === 'error') {
    runtimeErrors.push(message.params.entry.text);
  }
});

function command(method, params = {}) {
  const id = nextCommandId;
  nextCommandId += 1;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const response = await command('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  });
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.exception?.description
      || response.exceptionDetails.text);
  }
  return response.result.value;
}

const snapshotSource = `(async () => {
  const graphics = globalThis.__hyperscopeGraphicsBackend;
  const authority = globalThis.__hyperscopeWebGpuLodAuthority;
  const residency = graphics?.refresh ? await graphics.refresh() : null;
  let renderer = {};
  const sameContext = globalThis.__hyperscopeSameContextLod || {};
  try {
    renderer = globalThis.__hyperscopeRuntimeDiagnostics
      ?.residentLodEdges?.() || {};
  } catch {
    // The first navigation snapshots may race the page's WASM initializer.
  }
  const params = new URLSearchParams(location.search);
  return {
    url: location.href,
    graphics: graphics ? {
      effective: graphics.effective,
      state: graphics.state,
      mode: graphics.renderMode,
    } : null,
    animationPlaying: document.getElementById('animate-toggle')
      ?.classList.contains('on') ?? null,
    animationTime: Number(document.getElementById('time')?.value ?? NaN),
    routeCamera: [params.get('rx'), params.get('ry'), params.get('rz')],
    inversionRadius: Number(document.getElementById('mr')?.value ?? NaN),
    authority: authority ? {
      active: Boolean(authority.effectiveActive),
      dispatches: Number(authority.dispatches || 0),
      fullSceneDispatches: Number(authority.fullSceneDispatches || 0),
      mismatches: [...(authority.mismatches || [])],
    } : null,
    residency: residency ? {
      modelFaces: Number(residency.modelFaces || 0),
      presentationFrames: Number(residency.presentationFrames || 0),
      framesSubmitted: Number(residency.framesSubmitted || 0),
      frameFailures: Number(residency.frameFailures || 0),
      deviceLodDispatches: Number(residency.deviceLodDispatches || 0),
      deviceLodFrames: Number(residency.deviceLodFrames || 0),
      visibilityUploads: Number(residency.visibilityUploads || 0),
      visibilityUploadBytes: Number(residency.visibilityUploadBytes || 0),
      frameTableUploads: Number(residency.frameTableUploads || 0),
      frameTableReuses: Number(residency.frameTableReuses || 0),
      classifierPoseUploads: Number(residency.classifierPoseUploads || 0),
      residentPoseUploads: Number(residency.residentPoseUploads || 0),
      residentPoseReuses: Number(residency.residentPoseReuses || 0),
      lastDeviceLodEpoch: residency.lastDeviceLodEpoch ?? null,
      lastFrameFailure: residency.lastFrameFailure ?? null,
      lastError: residency.lastError ?? null,
    } : null,
    incumbent: {
      renderCalls: Number(renderer.renderCalls || 0),
      webglPatchFrames: Number(renderer.webglPatchFrames || 0),
      patchPrepareFrames: Number(renderer.patchPrepareFrames || 0),
      patchVisibilityFrames: Number(renderer.patchVisibilityFrames || 0),
    },
    sameContext: {
      state: sameContext.state,
      dispatches: Number(sameContext.dispatches || 0),
      completions: Number(sameContext.completions || 0),
      lastReadbackBytes: Number(sameContext.last_readback_bytes || 0),
      readbackBufferCreations: Number(sameContext.readback_buffer_creations || 0),
      readbackVectorCreations: Number(sameContext.readback_vector_creations || 0),
      failures: Number(sameContext.failures || 0),
      lastError: sameContext.last_error ?? null,
    },
  };
})()`;

async function snapshot() {
  return evaluate(snapshotSource);
}

async function waitFor(label, predicate, timeoutMs = 45_000) {
  const deadline = Date.now() + timeoutMs;
  let current = null;
  while (Date.now() < deadline) {
    current = await snapshot();
    if (predicate(current)) return current;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`${label} did not settle: ${JSON.stringify(current)}`);
}

function assertNoIncumbentOrReadbackGrowth(before, after, label) {
  assert.ok(after.incumbent.renderCalls > before.incumbent.renderCalls,
    `${label}: the shared frame loop did not advance: `
      + `${JSON.stringify({ before: before.incumbent, after: after.incumbent })}`);
  assert.equal(after.incumbent.webglPatchFrames, before.incumbent.webglPatchFrames,
    `${label}: WebGL submitted duplicate patch work`);
  assert.equal(after.incumbent.patchPrepareFrames, before.incumbent.patchPrepareFrames,
    `${label}: incumbent patch preparation advanced`);
  assert.equal(after.incumbent.patchVisibilityFrames, before.incumbent.patchVisibilityFrames,
    `${label}: incumbent patch visibility advanced`);
  assert.equal(after.sameContext.dispatches, before.sameContext.dispatches,
    `${label}: same-context WebGL LOD dispatched`);
  assert.equal(after.sameContext.completions, before.sameContext.completions,
    `${label}: same-context WebGL LOD completed a readback`);
  assert.equal(after.sameContext.lastReadbackBytes, before.sameContext.lastReadbackBytes,
    `${label}: same-context WebGL readback bytes changed`);
  assert.equal(after.sameContext.readbackBufferCreations,
    before.sameContext.readbackBufferCreations,
    `${label}: a WebGL readback buffer was allocated`);
  assert.equal(after.sameContext.readbackVectorCreations,
    before.sameContext.readbackVectorCreations,
    `${label}: a CPU readback vector was allocated`);
  assert.equal(after.residency.visibilityUploads, before.residency.visibilityUploads,
    `${label}: CPU visibility was uploaded into the device-resident path`);
  assert.equal(after.residency.visibilityUploadBytes, before.residency.visibilityUploadBytes,
    `${label}: CPU visibility upload bytes changed`);
}

try {
  await command('Runtime.enable');
  await command('Log.enable');
  await command('Page.enable');
  await command('Emulation.setDeviceMetricsOverride', {
    width: 960,
    height: 720,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await command('Page.navigate', { url: route.href });

  const ready = await waitFor('animated inverted WebGPU presentation', current =>
    current.graphics?.effective === 'webgpu'
      && current.graphics?.state === 'presenting'
      && current.graphics?.mode === 'wire'
      && current.animationPlaying === true
      && current.authority?.active === true
      && current.residency?.modelFaces === 984
      && current.residency?.presentationFrames >= 3
      && current.residency?.deviceLodDispatches >= 2
      && current.residency?.frameFailures === 0);

  await new Promise(resolve => setTimeout(resolve, 500));
  const animatedBefore = await snapshot();
  await new Promise(resolve => setTimeout(resolve, 800));
  const animatedAfter = await snapshot();
  assert.ok(animatedAfter.residency.presentationFrames
    > animatedBefore.residency.presentationFrames,
  'animation did not present new WebGPU frames');
  assert.ok(animatedAfter.residency.deviceLodDispatches
    > animatedBefore.residency.deviceLodDispatches,
  'animation did not refresh device-resident LOD');
  assert.ok(animatedAfter.residency.classifierPoseUploads
    > animatedBefore.residency.classifierPoseUploads,
  'animation did not publish classifier pose state');
  assert.equal(animatedAfter.residency.residentPoseUploads,
    animatedBefore.residency.residentPoseUploads,
  'renderer duplicated the classifier pose publication');
  assert.ok(animatedAfter.residency.residentPoseReuses
    > animatedBefore.residency.residentPoseReuses,
  'renderer did not reuse the classifier pose publication');
  assertNoIncumbentOrReadbackGrowth(animatedBefore, animatedAfter, 'animation');

  await evaluate(`document.getElementById('animate-toggle').click()`);
  const paused = await waitFor('paused animation', current =>
    current.animationPlaying === false
      && current.residency.presentationFrames >= animatedAfter.residency.presentationFrames);

  const canvasRect = await evaluate(`(() => {
    const rect = document.getElementById('cv').getBoundingClientRect();
    return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
  })()`);
  const start = {
    x: canvasRect.x + canvasRect.width * 0.5,
    y: canvasRect.y + canvasRect.height * 0.5,
  };
  await command('Input.dispatchMouseEvent', {
    type: 'mousePressed', x: start.x, y: start.y, button: 'left', buttons: 1,
    clickCount: 1,
  });
  for (const [dx, dy] of [[35, -12], [70, -25], [105, -38]]) {
    await command('Input.dispatchMouseEvent', {
      type: 'mouseMoved', x: start.x + dx, y: start.y + dy,
      button: 'none', buttons: 1,
    });
  }
  await command('Input.dispatchMouseEvent', {
    type: 'mouseReleased', x: start.x + 105, y: start.y - 38,
    button: 'left', buttons: 0, clickCount: 1,
  });
  const cameraAfter = await waitFor('camera motion publication', current =>
    current.routeCamera.some((value, index) => value !== paused.routeCamera[index])
      && current.residency.presentationFrames > paused.residency.presentationFrames
      && current.residency.deviceLodDispatches > paused.residency.deviceLodDispatches);
  assertNoIncumbentOrReadbackGrowth(paused, cameraAfter, 'camera motion');

  const requestedRadius = await evaluate(`(() => {
    const input = document.getElementById('mr');
    const next = Math.max(0.11, Number(input.value) * 0.72);
    input.value = String(next);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    return next;
  })()`);
  const inversionAfter = await waitFor('inversion-radius publication', current =>
    Math.abs(current.inversionRadius - requestedRadius) < 1e-6
      && current.residency.presentationFrames > cameraAfter.residency.presentationFrames
      && current.residency.deviceLodDispatches > cameraAfter.residency.deviceLodDispatches);
  assertNoIncumbentOrReadbackGrowth(cameraAfter, inversionAfter, 'inversion motion');

  for (const snapshotValue of [ready, animatedAfter, paused, cameraAfter, inversionAfter]) {
    assert.equal(snapshotValue.residency.frameFailures, 0);
    assert.equal(snapshotValue.residency.lastFrameFailure, null);
    assert.equal(snapshotValue.residency.lastError, null);
    assert.equal(snapshotValue.sameContext.failures, 0);
    assert.equal(snapshotValue.sameContext.lastError, null);
    assert.deepEqual(snapshotValue.authority.mismatches, []);
  }
  assert.deepEqual(runtimeErrors, []);

  console.log(JSON.stringify({
    route: route.href,
    ready,
    animated: { before: animatedBefore, after: animatedAfter },
    paused,
    cameraAfter,
    inversionAfter,
    runtimeErrors,
  }, null, 2));
} finally {
  await activate(originalPage.id);
  socket.close();
  await fetch(`${cdpEndpoint}/json/close/${page.id}`);
}
