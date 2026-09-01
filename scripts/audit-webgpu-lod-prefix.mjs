#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const targets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const originalPage = targets.find(target =>
  target.type === 'page' && target.url.includes(`:${pagePort}/`));
assert.ok(originalPage, 'a pre-existing Hyperscope page is required');

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
  const presentation = globalThis.__hyperscopePresentation;
  const authority = globalThis.__hyperscopeWebGpuLodAuthority;
  const app = globalThis.__hyperscopeAppShadowDiagnostics;
  const lod = globalThis.__hyperscopeRuntimeDiagnostics?.lod;
  const residency = graphics?.refresh ? await graphics.refresh() : null;
  return {
    graphics: graphics ? {
      effective: graphics.effective,
      state: graphics.state,
      error: graphics.error ?? null,
    } : null,
    presentation: presentation ? {
      ready: presentation.ready,
      assetsReady: presentation.assetsReady,
      activeCue: presentation.activeCue,
      packedFaces: Number(presentation.packedFaces || 0),
      residentAssets: Number(presentation.residentAssets?.length || 0),
      desiredAssets: Number(presentation.desiredAssets?.length || 0),
      error: presentation.error ?? null,
    } : null,
    authority: authority ? {
      active: Boolean(authority.effectiveActive),
      dispatches: Number(authority.dispatches || 0),
      fullSceneDispatches: Number(authority.fullSceneDispatches || 0),
      mismatches: [...(authority.mismatches || [])],
    } : null,
    residency: residency ? {
      modelFaces: Number(residency.modelFaces || 0),
      presentationFrames: Number(residency.presentationFrames || 0),
      deviceLodDispatches: Number(residency.deviceLodDispatches || 0),
      deviceLodFullDispatches: Number(residency.deviceLodFullDispatches || 0),
      deviceLodPrefixDispatches: Number(residency.deviceLodPrefixDispatches || 0),
      lastDeviceLodClassifiedFaces: Number(residency.lastDeviceLodClassifiedFaces || 0),
      frameFailures: Number(residency.frameFailures || 0),
      lastFrameFailure: residency.lastFrameFailure ?? null,
      lastError: residency.lastError ?? null,
    } : null,
    appFrameErrors: Number(app?.frameErrors || 0),
    lodErrors: Number(lod?.errors || 0),
    lodPoseMismatches: Number(lod?.poseMismatches || 0),
  };
})()`;

async function snapshot() {
  return evaluate(snapshotSource);
}

async function waitFor(label, predicate, attempts = 900) {
  let current = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    current = await snapshot();
    if (predicate(current)) return current;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`${label} did not settle: ${JSON.stringify(current)}`);
}

const route = new URL(`http://127.0.0.1:${pagePort}/`);
for (const [key, value] of [
  ['presentation', '1'],
  ['cue', 'e0000000-0000-4000-8000-000000000002'],
  ['glb', 'horse.glb'],
  ['presentimpl', 'rust'],
  ['sceneimpl', 'rust'],
  ['animclipimpl', 'rust'],
  ['animclockimpl', 'rust'],
  ['gfx', 'webgpu'],
  ['lodimpl', 'rust'],
]) route.searchParams.set(key, value);

try {
  await command('Runtime.enable');
  await command('Log.enable');
  await command('Page.enable');
  await command('Page.navigate', { url: route.href });

  const readyPrefix = await waitFor('composed WebGPU prefix epoch', current =>
    current.graphics?.effective === 'webgpu'
      && current.presentation?.ready === true
      && current.presentation?.assetsReady === true
      && current.presentation?.activeCue === 'e0000000-0000-4000-8000-000000000002'
      && current.presentation?.packedFaces > 984
      && current.authority?.active === true
      && current.residency?.modelFaces === current.presentation?.packedFaces
      && current.residency?.deviceLodFullDispatches >= 1
      && current.residency?.deviceLodPrefixDispatches >= 2
      && current.residency?.lastDeviceLodClassifiedFaces > 0
      && current.residency?.lastDeviceLodClassifiedFaces
        < current.residency?.modelFaces);

  // The final cue has a 1.2-second camera/layer transition. Complete scene
  // epochs during that interval are required; measure animation-only traffic
  // only after the authored transition has settled.
  await new Promise(resolve => setTimeout(resolve, 1800));
  const before = await snapshot();
  await new Promise(resolve => setTimeout(resolve, 1000));
  const after = await snapshot();
  const audit = { targetId: page.id, readyPrefix, before, after, runtimeErrors };
  console.log(JSON.stringify(audit, null, 2));

  assert.equal(after.presentation.error, null);
  assert.ok(after.presentation.residentAssets >= after.presentation.desiredAssets);
  assert.equal(after.residency.modelFaces, after.presentation.packedFaces);
  assert.ok(after.residency.lastDeviceLodClassifiedFaces < after.residency.modelFaces);
  assert.ok(after.residency.deviceLodPrefixDispatches
    > before.residency.deviceLodPrefixDispatches,
  'animation did not advance topology-closed prefix classifications');
  assert.equal(after.residency.deviceLodFullDispatches,
    before.residency.deviceLodFullDispatches,
  'settled animation unnecessarily reclassified the static suffix');
  assert.equal(after.authority.fullSceneDispatches,
    before.authority.fullSceneDispatches,
  'authority policy unnecessarily requested a complete animation epoch');
  assert.equal(after.authority.mismatches.length, 0);
  assert.equal(after.residency.frameFailures, 0);
  assert.equal(after.residency.lastFrameFailure, null);
  assert.equal(after.residency.lastError, null);
  assert.equal(after.appFrameErrors, 0);
  assert.equal(after.lodErrors, 0);
  assert.equal(after.lodPoseMismatches, 0);
  assert.deepEqual(runtimeErrors, []);
} finally {
  await activate(originalPage.id);
  socket.close();
  await fetch(`${cdpEndpoint}/json/close/${page.id}`);
}
