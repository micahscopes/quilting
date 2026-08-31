#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const targetId = process.env.HYPERSCOPE_TARGET_ID || null;
const expectedImplementation = process.env.HYPERSCOPE_EXPECT_ANIMCLOCK || 'rust';
const routeImplementation = process.env.HYPERSCOPE_ANIMCLOCK_ROUTE || expectedImplementation;
assert.ok(['js', 'shadow', 'rust'].includes(expectedImplementation),
  'HYPERSCOPE_EXPECT_ANIMCLOCK must be js, shadow, or rust');
assert.ok(['implicit', 'js', 'shadow', 'rust'].includes(routeImplementation),
  'HYPERSCOPE_ANIMCLOCK_ROUTE must be implicit, js, shadow, or rust');

const beforeTargets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const originalPage = beforeTargets.find(target => target.type === 'page'
  && target.url.includes(`:${pagePort}/`)
  && target.id !== targetId);
const ownsTarget = targetId == null;
const route = new URL(`http://127.0.0.1:${pagePort}/`);
for (const [key, value] of [
  ['glb', 'ant.glb'],
  ['gfx', 'webgl2'],
  ['animate', '1'],
  ['anim', '0'],
  ['animtime', '3'],
  ['animspeed', '-1'],
]) {
  route.searchParams.set(key, value);
}
if (routeImplementation !== 'implicit') {
  route.searchParams.set('animclockimpl', routeImplementation);
}
const page = ownsTarget
  ? await (await fetch(
    `${cdpEndpoint}/json/new?${encodeURIComponent(route.href)}`,
    { method: 'PUT' },
  )).json()
  : beforeTargets.find(target => target.type === 'page' && target.id === targetId);
assert.ok(page, `animation-clock audit target ${targetId || '(new)'} was not found`);

async function activate(id) {
  const response = await fetch(`${cdpEndpoint}/json/activate/${id}`);
  if (!response.ok) throw new Error(`could not activate Chrome target ${id}`);
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
  const sample = new Float64Array(4);
  let sampleError = null;
  try { app?.writeInstalledAnimationSample(sample); }
  catch (error) { sampleError = error?.message || String(error); }
  const runtime = app?.animationRuntimeState?.();
  const active = runtime?.clipState?.active?.clip;
  const clock = globalThis.__hyperscopeAnimationClockDiagnostics;
  const pose = globalThis.__hyperscopeAnimationPoseDiagnostics;
  const appDiagnostics = globalThis.__hyperscopeAppShadowDiagnostics;
  const lod = globalThis.__hyperscopeRuntimeDiagnostics?.lod;
  const rustTimeline = document.getElementById('animation-time-rust');
  const browserTimeline = document.getElementById('animation-time-browser');
  const rustInput = document.getElementById('animation-time-rust-input');
  const browserInput = document.getElementById('time');
  const params = new URLSearchParams(location.search);
  return {
    visibility: document.visibilityState,
    stats: document.getElementById('stats')?.textContent || '',
    sample: sampleError ? null : Array.from(sample),
    sampleError,
    runtime: runtime ? {
      playing: runtime.playing,
      timeSeconds: runtime.timeSeconds,
      speed: runtime.speed,
      clip: active ? {
        index: active.index,
        name: active.name,
        timeMinSeconds: active.timeMinSeconds,
        timeMaxSeconds: active.timeMaxSeconds,
      } : null,
    } : null,
    clock: clock ? { ...clock } : null,
    pose: pose ? {
      ...pose,
      lastAppliedRequest: pose.lastAppliedRequest ? { ...pose.lastAppliedRequest } : null,
      lastRejectedRequest: pose.lastRejectedRequest ? { ...pose.lastRejectedRequest } : null,
    } : null,
    app: appDiagnostics ? {
      frameCalls: Number(appDiagnostics.frameCalls || 0),
      maximumFrameDeltaSeconds: Number(appDiagnostics.maximumFrameDeltaSeconds || 0),
      frameErrors: Number(appDiagnostics.frameErrors || 0),
      animationPoseMismatches: (appDiagnostics.mismatches || [])
        .filter(mismatch => mismatch.code === 'animation_pose_scheduler')
        .map(mismatch => ({ ...mismatch })),
    } : null,
    lod: lod ? {
      submitted: Number(lod.submitted || 0),
      completed: Number(lod.completed || 0),
      poseSubmittedRevision: Number(lod.poseSubmittedRevision || 0),
      poseCompletedRevision: Number(lod.poseCompletedRevision || 0),
      poseContinuityEpoch: Number(lod.poseContinuityEpoch || 0),
      poseLagRevisions: Number(lod.poseLagRevisions || 0),
      poseMismatches: Number(lod.poseMismatches || 0),
      poseRetiredPublications: Number(lod.poseRetiredPublications || 0),
      errors: Number(lod.errors || 0),
    } : null,
    timeline: {
      rustHidden: rustTimeline?.hidden ?? null,
      browserHidden: browserTimeline?.hidden ?? null,
      rustValue: rustInput?.value ?? null,
      browserValue: browserInput?.value ?? null,
    },
    url: {
      implementation: params.get('animclockimpl'),
      playing: params.get('animate'),
      time: params.get('animtime'),
      speed: params.get('animspeed'),
    },
  };
})()`;

async function snapshot() {
  return evaluate(snapshotSource);
}

async function waitFor(label, predicate, attempts = 300) {
  let current = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    current = await snapshot();
    if (predicate(current)) return current;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`${label} did not settle: ${JSON.stringify(current)}`);
}

async function togglePlayback() {
  await evaluate(`(() => {
    const button = document.querySelector('#animation-control-rust-view button');
    if (!button) throw new Error('Rust playback control is unavailable');
    button.click();
  })()`);
}

async function seekTimeline(seconds) {
  await evaluate(`(() => {
    const implementation = ${JSON.stringify(expectedImplementation)};
    const input = document.getElementById(
      implementation === 'rust' ? 'animation-time-rust-input' : 'time',
    );
    if (!input) throw new Error('animation timeline is unavailable');
    input.value = ${JSON.stringify(String(seconds))};
    input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
  })()`);
}

let audit;
try {
  await command('Runtime.enable');
  await command('Log.enable');
  const initial = await waitFor('long reverse animation startup', current => {
    const expectedState = expectedImplementation === 'rust'
      ? 'authoritative' : expectedImplementation === 'shadow' ? 'shadowing' : 'browser';
    return current.visibility === 'visible'
      && current.stats.includes('ant.glb')
      && current.runtime?.clip?.name === 'Take 001'
      && Math.abs(current.runtime.clip.timeMaxSeconds - 75.80000305175781) < 1e-9
      && current.runtime.playing === true
      && current.runtime.speed === -1
      && current.sample?.[0] === 1
      && current.sample?.[1] === 0
      && current.sample?.[3] === -1
      && current.clock?.implementation === expectedImplementation
      && current.clock?.state === expectedState
      && current.clock?.routeRestores >= 1
      && current.pose?.lastAppliedRequest != null;
  });

  const wrapped = await waitFor('reverse long-clip wrap', current =>
    current.runtime?.timeSeconds < 0
      && current.sample?.[2] > 70
      && current.pose?.lastAppliedRequest?.t > 70
      && (expectedImplementation === 'js'
        || current.pose?.rebases > initial.pose.rebases)
      && current.clock?.errors === 0
      && current.pose?.errors === 0);

  assert.ok(originalPage, 'a pre-existing Hyperscope page is required for background cadence');
  const beforeBackground = await snapshot();
  await activate(originalPage.id);
  const hidden = await waitFor('background visibility', current =>
    current.visibility === 'hidden');
  const hiddenStartedAt = performance.now();
  await new Promise(resolve => setTimeout(resolve, 3200));
  const background = await snapshot();
  const backgroundWallSeconds = (performance.now() - hiddenStartedAt) / 1000;
  const backgroundFrameCalls = background.app.frameCalls - hidden.app.frameCalls;
  assert.equal(background.visibility, 'hidden');
  assert.ok(backgroundFrameCalls <= 8,
    `hidden audit was not throttled: ${backgroundFrameCalls} frames`);

  await activate(page.id);
  const resumed = await waitFor('foreground clock resumption', current =>
    current.visibility === 'visible'
      && current.app.frameCalls > background.app.frameCalls
      && current.app.maximumFrameDeltaSeconds >= 0.24
      && current.pose.lastAppliedRequest?.revision
        > (background.pose.lastAppliedRequest?.revision || 0));

  await togglePlayback();
  const paused = await waitFor('paused Rust playback', current =>
    current.runtime?.playing === false
      && current.sample?.[0] === 0
      && current.url.playing === '0');

  await seekTimeline(37.9);
  const sought = await waitFor('paused timeline seek', current => {
    const visibleValue = Number(expectedImplementation === 'rust'
      ? current.timeline.rustValue : current.timeline.browserValue);
    const rendererAtSeek = Math.abs(current.pose?.lastAppliedRequest?.t - 37.9) < 1e-9;
    const rustClockAtSeek = expectedImplementation === 'js'
      || (Math.abs(current.runtime?.timeSeconds - 37.9) < 1e-9
        && Math.abs(current.sample?.[2] - 37.9) < 1e-9);
    const urlAtSeek = expectedImplementation !== 'rust'
      || Math.abs(Number(current.url.time) - 37.9) < 1e-9;
    return Math.abs(visibleValue - 37.9) < 1e-9
      && rendererAtSeek && rustClockAtSeek && urlAtSeek;
  });

  await togglePlayback();
  const resumedPlayback = await waitFor('resumed reverse playback', current =>
    current.runtime?.playing === true
      && current.sample?.[0] === 1
      && current.url.playing == null
      && Number(current.pose?.lastAppliedRequest?.t) < 37.85);

  const final = await snapshot();
  audit = {
    targetId: page.id,
    initial,
    wrapped,
    beforeBackground,
    background: {
      ...background,
      wallSeconds: backgroundWallSeconds,
      frameCalls: backgroundFrameCalls,
    },
    resumed,
    paused,
    sought,
    resumedPlayback,
    final,
    runtimeErrors,
  };

  console.log(JSON.stringify(audit, null, 2));

  assert.equal(final.clock.errors, 0);
  assert.equal(final.clock.mismatches, 0);
  assert.equal(final.pose.errors, 0);
  assert.equal(final.pose.mismatches, 0);
  assert.equal(final.app.frameErrors, 0);
  assert.equal(final.lod.poseMismatches, 0);
  assert.equal(final.lod.errors, 0);
  assert.ok(final.lod.poseCompletedRevision > 0);
  assert.deepEqual(runtimeErrors, []);
  assert.equal(
    initial.url.implementation,
    routeImplementation === 'implicit' ? null : routeImplementation,
  );
  if (expectedImplementation === 'rust') {
    assert.equal(final.clock.fallbackWrites, 0);
    assert.ok(final.clock.authorityWrites > initial.clock.authorityWrites);
    assert.equal(final.clock.state, 'authoritative');
    assert.equal(final.pose.state, 'authoritative');
    assert.equal(initial.timeline.rustHidden, false);
    assert.equal(initial.timeline.browserHidden, true);
  } else if (expectedImplementation === 'shadow') {
    assert.ok(final.clock.comparisons > initial.clock.comparisons);
    assert.ok(final.pose.comparisons > initial.pose.comparisons);
    assert.ok(final.clock.maximumError <= 2e-6);
    assert.equal(final.clock.authorityWrites, 0);
    assert.equal(final.clock.fallbackWrites, 0);
  } else {
    for (const counter of [
      'synchronizations', 'dispatches', 'samples', 'comparisons', 'mismatches',
      'authorityWrites', 'fallbackWrites', 'errors',
    ]) {
      assert.equal(final.clock[counter], 0, `JavaScript clock changed ${counter}`);
    }
    assert.equal(final.pose.state, 'browser');
  }
} finally {
  if (originalPage) await activate(originalPage.id);
  socket.close();
  if (ownsTarget) await fetch(`${cdpEndpoint}/json/close/${page.id}`);
}
