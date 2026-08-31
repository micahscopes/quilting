#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const expectedImplementation = process.env.HYPERSCOPE_EXPECT_PATCHLAB || 'rust';
const routeImplementation = process.env.HYPERSCOPE_PATCHLAB_ROUTE || expectedImplementation;
const initialRatio = Number(process.env.HYPERSCOPE_PATCHLAB_RATIO || 2);
assert.ok(['js', 'shadow', 'rust'].includes(expectedImplementation),
  'HYPERSCOPE_EXPECT_PATCHLAB must be js, shadow, or rust');
assert.ok(['implicit', 'js', 'shadow', 'rust'].includes(routeImplementation),
  'HYPERSCOPE_PATCHLAB_ROUTE must be implicit, js, shadow, or rust');
assert.ok([2, 4].includes(initialRatio), 'HYPERSCOPE_PATCHLAB_RATIO must be 2 or 4');

const beforeTargets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const originalPage = beforeTargets.find(target => target.type === 'page'
  && target.url.includes(`:${pagePort}/`));
const route = new URL(`http://127.0.0.1:${pagePort}/`);
route.searchParams.set('_audit', String(Date.now()));
for (const [key, value] of [
  ['gfx', 'webgl2'],
  ['lab', 'triangle'],
  ['labfield', 'edges'],
  ['laba', '1'],
  ['labb', '6'],
  ['labc', '6'],
  ['atlas', '7'],
  ['lodratio', String(initialRatio)],
  ['animate', '0'],
]) route.searchParams.set(key, value);
if (routeImplementation !== 'implicit') {
  route.searchParams.set('patchlabimpl', routeImplementation);
}
const page = await (await fetch(
  `${cdpEndpoint}/json/new?${encodeURIComponent(route.href)}`,
  { method: 'PUT' },
)).json();

async function activate(id) {
  if (!id) return;
  const response = await fetch(`${cdpEndpoint}/json/activate/${id}`);
  if (!response.ok) throw new Error(`could not activate Chrome target ${id}`);
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

await command('Runtime.enable');
await command('Log.enable');

const snapshotSource = `(() => {
  const diagnostics = globalThis.__hyperscopePatchLabDiagnostics;
  const app = globalThis.__hyperscopeAppShadowController;
  const state = app?.patchLabSnapshot?.() ?? null;
  const lab = globalThis.__hyperscopePatchLab;
  const atlas = globalThis.__hyperscopeResidentAtlas;
  const params = new URLSearchParams(location.search);
  return {
    url: location.href,
    route: {
      implementation: params.get('patchlabimpl'),
      lab: params.get('lab'),
      field: params.get('labfield'),
      edges: [params.get('laba'), params.get('labb'), params.get('labc')],
      atlas: params.get('atlas'),
      ratio: params.get('lodratio'),
      animate: params.get('labanimate'),
    },
    diagnostics: diagnostics ? {
      implementation: diagnostics.implementation,
      state: diagnostics.state,
      synchronizations: Number(diagnostics.synchronizations || 0),
      commits: Number(diagnostics.commits || 0),
      effects: Number(diagnostics.effects || 0),
      hostedEffects: Number(diagnostics.hostedEffects || 0),
      completedEffects: Number(diagnostics.completedEffects || 0),
      cancelledEffects: Number(diagnostics.cancelledEffects || 0),
      preventedStaleInstalls: Number(diagnostics.preventedStaleInstalls || 0),
      geometryCompletions: Number(diagnostics.geometryCompletions || 0),
      lodCompletions: Number(diagnostics.lodCompletions || 0),
      staleCompletions: Number(diagnostics.staleCompletions || 0),
      comparisons: Number(diagnostics.comparisons || 0),
      mismatches: Number(diagnostics.mismatches || 0),
      errors: Number(diagnostics.errors || 0),
      effectHostErrors: Number(diagnostics.effectHostErrors || 0),
      viewAuthority: diagnostics.viewAuthority,
      effectHostAuthority: diagnostics.effectHostAuthority,
      viewMountErrors: Number(diagnostics.viewMountErrors || 0),
    } : null,
    state,
    browser: lab ? {
      active: lab.active,
      shape: lab.shape,
      animate: lab.animate,
      updateInFlight: lab.updateInFlight,
      updateDirty: lab.updateDirty,
      lastRequest: lab.lastRequest ? { ...lab.lastRequest } : null,
      lastResult: lab.lastResult ? {
        requestedFirstFace: lab.lastResult.requested?.slice(0, 3) ?? null,
        residentFirstFace: lab.lastResult.actual?.slice(0, 3) ?? null,
        requestedValues: lab.lastResult.requested?.length ?? 0,
        residentValues: lab.lastResult.actual?.length ?? 0,
        promotedEdges: lab.lastResult.promotedEdges,
        residentTriangles: lab.lastResult.residentTriangles,
        mismatchedEdges: lab.lastResult.mismatchedEdges,
        policyRatio: lab.lastResult.policyRatio,
        policyAligned: lab.lastResult.policyAligned,
      } : null,
    } : null,
    atlas: atlas ? {
      requestedExponent: atlas.requestedExponent,
      requestedGradingRatio: atlas.requestedGradingRatio,
      residentExponent: atlas.residentExponent,
      residentGradingRatio: atlas.residentGradingRatio,
      inFlight: atlas.inFlight,
      builds: atlas.builds,
      skippedStaleBuilds: atlas.skippedStaleBuilds,
      error: atlas.error,
    } : null,
    view: {
      rustHidden: document.getElementById('patch-lab-rust')?.hidden,
      browserHidden: document.getElementById('patch-lab-browser')?.hidden,
      rustInputs: document.querySelectorAll('#patch-lab-rust-view input').length,
      rustButtons: document.querySelectorAll('#patch-lab-rust-view button').length,
    },
    stats: document.getElementById('patch-lab-stats')?.textContent || '',
  };
})()`;

async function snapshot() {
  return evaluate(snapshotSource);
}

async function waitFor(label, predicate, timeoutMs = 25_000) {
  const deadline = Date.now() + timeoutMs;
  let current = null;
  while (Date.now() < deadline) {
    current = await snapshot();
    if (predicate(current)) return current;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`${label} did not settle: ${JSON.stringify(current)}`);
}

const settled = current => current.browser?.active
  && current.state?.installedGeometry
  && current.state?.latestLod
  && current.state.pendingGeometryJob == null
  && current.state.pendingLodJob == null
  && !current.atlas?.inFlight;

let audit;
try {
  const before = await waitFor('initial Patch Lab', current => {
    if (expectedImplementation === 'rust') return settled(current);
    return current.browser?.active && current.browser?.lastResult && !current.browser.updateInFlight;
  });
  if (expectedImplementation === 'rust') {
    const setRustInput = async (label, value, event = 'input') => evaluate(`(() => {
      const input = document.querySelector('#patch-lab-rust-view input[aria-label="${label}"]');
      if (!input) throw new Error('missing Rust Patch Lab input: ${label}');
      input.value = '${value}';
      input.dispatchEvent(new Event('${event}', { bubbles: true, composed: true }));
    })()`);
    await setRustInput('BC requested exponent', 3);
    await setRustInput('BC requested exponent', 5);
    await setRustInput('BC requested exponent', 1);
    const rapidEdges = await waitFor('coalesced Rust edge controls', current => settled(current)
      && current.state.controls.manualEdgeExponents.join(',') === '1,6,6'
      && current.browser.lastRequest?.edgeAExp === 1);

    await evaluate(`(() => {
      const button = [...document.querySelectorAll('#render-controls-rust-view button')]
        .find(candidate => candidate.textContent.trim() === '4:1');
      if (!button) throw new Error('missing Rust 4:1 grading control');
      button.click();
    })()`);
    const fourToOne = await waitFor('live 4:1 resident policy', current => settled(current)
      && current.atlas.residentGradingRatio === 4
      && current.browser.lastResult?.policyRatio === 4
      && current.state.latestLod?.residentFirstFace?.join(',') === '16,64,64');

    await evaluate(`(() => {
      const ranges = [...document.querySelectorAll('#render-controls-rust-view input[type="range"]')];
      const atlas = ranges.at(-1);
      if (!atlas) throw new Error('missing Rust atlas control');
      atlas.value = '6';
      atlas.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
    })()`);
    const atlasSix = await waitFor('debounced live atlas replacement', current => settled(current)
      && current.atlas.residentExponent === 6
      && current.state.controls.maxExponent <= 6
      && current.route.atlas === '6');

    const beforeAnimationCompletions = atlasSix.diagnostics.lodCompletions;
    const beforePhase = atlasSix.state.controls.phaseMicroradians;
    await evaluate(`(() => {
      const input = document.querySelector('#patch-lab-rust-view input[type="checkbox"]');
      if (!input) throw new Error('missing Rust Patch Lab animation control');
      input.checked = true;
      input.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
    })()`);
    const animated = await waitFor('Rust Patch Lab animation', current => settled(current)
      && current.state.controls.animate
      && current.state.controls.phaseMicroradians !== beforePhase
      && current.diagnostics.lodCompletions > beforeAnimationCompletions);
    await evaluate(`(() => {
      const input = document.querySelector('#patch-lab-rust-view input[type="checkbox"]');
      input.checked = false;
      input.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
    })()`);
    const paused = await waitFor('paused Rust Patch Lab animation', current => settled(current)
      && !current.state.controls.animate);

    await evaluate(`(() => {
      const button = [...document.querySelectorAll('#patch-lab-rust-view button')]
        .find(candidate => candidate.textContent.trim() === 'Plane');
      if (!button) throw new Error('missing Rust Plane control');
      button.click();
    })()`);
    const plane = await waitFor('Rust plane geometry', current => settled(current)
      && current.state.controls.shape === 'plane'
      && current.browser.shape === 'plane');
    await setRustInput('Plane grid width', 10, 'change');
    const grid = await waitFor('Rust plane grid geometry', current => settled(current)
      && current.state.controls.grid === 10
      && current.state.installedGeometry.grid === 10);
    audit = { before, rapidEdges, fourToOne, atlasSix, animated, paused, plane, grid };
  } else {
    const original = Number((await evaluate(
      `document.getElementById('patch-lab-edge-a').value`,
    )));
    const next = original === 2 ? 3 : 2;
    await evaluate(`(() => {
      const input = document.getElementById('patch-lab-edge-a');
      input.value = '${next}';
      input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
    })()`);
    const changed = await waitFor('browser Patch Lab edge control', current =>
      current.browser?.lastRequest?.edgeAExp === next && !current.browser.updateInFlight);
    audit = { before, changed };
  }
} catch (error) {
  error.message += `\nRuntime errors:\n${runtimeErrors.join('\n') || '(none captured)'}`;
  throw error;
} finally {
  socket.close();
  await fetch(`${cdpEndpoint}/json/close/${page.id}`);
  await activate(originalPage?.id);
}

console.log(JSON.stringify(audit, null, 2));
assert.equal(runtimeErrors.length, 0, runtimeErrors.join('\n'));
assert.equal(audit.before.diagnostics.implementation, expectedImplementation);
assert.equal(audit.before.diagnostics.mismatches, 0);
assert.equal(audit.before.diagnostics.errors, 0);
assert.equal(audit.before.diagnostics.effectHostErrors, 0);
if (expectedImplementation === 'rust') {
  assert.equal(audit.before.diagnostics.viewAuthority, 'hyperscope-web');
  assert.equal(audit.before.diagnostics.effectHostAuthority, 'browser-io-adapter');
  assert.equal(audit.before.view.rustHidden, false);
  assert.equal(audit.before.view.browserHidden, true);
  assert.equal(audit.grid.diagnostics.mismatches, 0);
  assert.equal(audit.grid.diagnostics.errors, 0);
  assert.equal(audit.grid.diagnostics.effectHostErrors, 0);
  assert.ok(audit.grid.diagnostics.preventedStaleInstalls > 0,
    'animated/coalesced edits did not exercise the stale renderer-install fence');
} else if (expectedImplementation === 'shadow') {
  assert.equal(audit.before.diagnostics.viewAuthority, 'hyperscope-web-shadow');
  assert.equal(audit.before.view.rustHidden, true);
  assert.equal(audit.before.view.browserHidden, false);
  assert.equal(audit.changed.diagnostics.mismatches, 0);
  assert.equal(audit.changed.diagnostics.errors, 0);
} else {
  assert.equal(audit.before.diagnostics.state, 'disabled');
  assert.equal(audit.before.view.rustHidden, true);
  assert.equal(audit.before.view.browserHidden, false);
  assert.equal(audit.changed.diagnostics.effects, 0);
}
