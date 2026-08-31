#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const beforeTargets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const originalPage = beforeTargets.find(target =>
  target.type === 'page' && target.url.includes(`:${pagePort}/`));
assert.ok(originalPage, 'a pre-existing Hyperscope page is required');

const route = new URL(`http://127.0.0.1:${pagePort}/`);
for (const [key, value] of [
  ['presentation', '1'],
  ['presentimpl', 'rust'],
  ['animclipimpl', 'rust'],
  ['animclockimpl', 'rust'],
  ['gfx', 'webgl2'],
]) {
  route.searchParams.set(key, value);
}
const page = await (await fetch(
  `${cdpEndpoint}/json/new?${encodeURIComponent(route.href)}`,
  { method: 'PUT' },
)).json();

async function activate(targetId) {
  const response = await fetch(`${cdpEndpoint}/json/activate/${targetId}`);
  if (!response.ok) throw new Error(`could not activate Chrome target ${targetId}`);
}

await activate(page.id);
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((open, reject) => {
  socket.addEventListener('open', open, { once: true });
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

const snapshotSource = `(() => {
  const app = globalThis.__hyperscopeAppShadowController;
  const presentation = globalThis.__hyperscopePresentation;
  const appPresentation = app?.snapshot?.()?.presentation?.active;
  const runtime = app?.animationRuntimeState?.();
  const sample = new Float64Array(4);
  let sampleError = null;
  try { app?.writeInstalledAnimationSample(sample); }
  catch (error) { sampleError = error?.message || String(error); }
  const clock = globalThis.__hyperscopeAnimationClockDiagnostics;
  const pose = globalThis.__hyperscopeAnimationPoseDiagnostics;
  const clip = globalThis.__hyperscopeAnimationClipDiagnostics;
  const appDiagnostics = globalThis.__hyperscopeAppShadowDiagnostics;
  const lod = globalThis.__hyperscopeRuntimeDiagnostics?.lod;
  return {
    stats: document.getElementById('stats')?.textContent || '',
    presentation: presentation ? {
      ready: presentation.ready,
      assetsReady: presentation.assetsReady,
      activeCue: presentation.activeCue,
      activeAnimation: presentation.activeAnimation
        ? { ...presentation.activeAnimation } : null,
      desiredAssets: presentation.desiredAssets?.length ?? null,
      residentAssets: presentation.residentAssets?.length ?? null,
      packedFaces: presentation.packedFaces,
      lodResidentFaces: presentation.lodResidentFaces,
      error: presentation.error,
    } : null,
    cue: appPresentation ? {
      index: appPresentation.cue_index,
      id: appPresentation.cue_id,
    } : null,
    runtime: runtime ? {
      playing: runtime.playing,
      timeSeconds: runtime.timeSeconds,
      speed: runtime.speed,
      activeClip: runtime.clipState?.active?.clip?.name ?? null,
      pendingClip: runtime.clipState?.pending?.clip?.name ?? null,
    } : null,
    sample: sampleError ? null : Array.from(sample),
    sampleError,
    clock: clock ? { ...clock } : null,
    pose: pose ? { ...pose } : null,
    clip: clip ? { ...clip } : null,
    app: appDiagnostics ? {
      frameErrors: Number(appDiagnostics.frameErrors || 0),
      mismatches: (appDiagnostics.mismatches || [])
        .filter(mismatch => String(mismatch.code).includes('animation'))
        .map(mismatch => ({ ...mismatch })),
    } : null,
    lod: lod ? {
      poseMismatches: Number(lod.poseMismatches || 0),
      poseLagRevisions: Number(lod.poseLagRevisions || 0),
      errors: Number(lod.errors || 0),
    } : null,
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

const cueIds = [
  'e0000000-0000-4000-8000-000000000007',
  'e0000000-0000-4000-8000-000000000008',
  'e0000000-0000-4000-8000-000000000009',
  'e0000000-0000-4000-8000-00000000000a',
  'e0000000-0000-4000-8000-000000000001',
];

const steps = [];
try {
  await command('Runtime.enable');
  await command('Log.enable');
  const initial = await waitFor('paused initial presentation cue', current =>
    current.presentation?.ready === true
      && current.presentation?.assetsReady === true
      && current.presentation?.error == null
      && current.cue?.index === 0
      && current.cue?.id === cueIds[0]
      && current.presentation?.activeAnimation?.playing === false
      && current.runtime?.playing === false
      && current.runtime?.activeClip === 'horse_A_'
      && current.runtime?.pendingClip == null
      && current.clock?.implementation === 'rust'
      && current.clock?.state === 'authoritative'
      && current.pose?.state === 'authoritative'
      && current.clip?.implementation === 'rust'
      && current.clip?.state === 'parity');
  steps.push(initial);

  for (let index = 1; index < cueIds.length; index += 1) {
    await evaluate('globalThis.__hyperscopePresentationAdvance()');
    const expectedPlaying = index === 4;
    steps.push(await waitFor(`presentation cue ${index + 1}`, current =>
      current.cue?.index === index
        && current.cue?.id === cueIds[index]
        && current.presentation?.activeCue === cueIds[index]
        && current.presentation?.activeAnimation?.playing === expectedPlaying
        && current.runtime?.playing === expectedPlaying
        && current.runtime?.pendingClip == null
        && current.presentation?.error == null));
  }

  const playing = steps.at(-1);
  await new Promise(resolve => setTimeout(resolve, 600));
  const advancedClock = await waitFor('playing cue clock advancement', current =>
    current.cue?.index === 4
      && current.runtime?.timeSeconds > playing.runtime.timeSeconds
      && current.sample?.[2] > playing.sample[2]
      && current.pose?.lastAppliedRequest?.t > playing.pose.lastAppliedRequest.t);

  await evaluate('globalThis.__hyperscopePresentationReverse()');
  const pausedAgain = await waitFor('reverse to paused cue', current =>
    current.cue?.index === 3
      && current.cue?.id === cueIds[3]
      && current.presentation?.activeAnimation?.playing === false
      && current.runtime?.playing === false
      && current.sample?.[0] === 0);
  const pausedTime = pausedAgain.runtime.timeSeconds;
  await new Promise(resolve => setTimeout(resolve, 500));
  const stablePause = await snapshot();
  assert.equal(stablePause.runtime.timeSeconds, pausedTime);

  const final = stablePause;
  const audit = { targetId: page.id, steps, advancedClock, pausedAgain, final, runtimeErrors };
  console.log(JSON.stringify(audit, null, 2));

  assert.equal(final.presentation.error, null);
  assert.ok(final.presentation.residentAssets >= final.presentation.desiredAssets);
  assert.equal(final.presentation.packedFaces, final.presentation.lodResidentFaces);
  assert.equal(final.clock.errors, 0);
  assert.equal(final.clock.mismatches, 0);
  assert.equal(final.clock.fallbackWrites, 0);
  assert.ok(final.clock.authorityWrites > initial.clock.authorityWrites);
  assert.equal(final.pose.errors, 0);
  assert.equal(final.pose.mismatches, 0);
  assert.equal(final.clip.errors, 0);
  assert.equal(final.clip.mismatches, 0);
  assert.equal(final.app.frameErrors, 0);
  assert.deepEqual(final.app.mismatches, []);
  assert.equal(final.lod.poseMismatches, 0);
  assert.equal(final.lod.errors, 0);
  assert.deepEqual(runtimeErrors, []);
} finally {
  await activate(originalPage.id);
  socket.close();
  await fetch(`${cdpEndpoint}/json/close/${page.id}`);
}
