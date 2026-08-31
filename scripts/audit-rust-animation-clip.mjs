#!/usr/bin/env node

import assert from 'node:assert/strict';
import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { basename, resolve } from 'node:path';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const targetId = process.env.HYPERSCOPE_TARGET_ID || null;
const expectedImplementation = process.env.HYPERSCOPE_EXPECT_ANIMCLIP || 'rust';
const routeImplementation = process.env.HYPERSCOPE_ANIMCLIP_ROUTE || expectedImplementation;
const fixturePath = resolve(process.env.HYPERSCOPE_ANIMATION_GLB
  || '/home/micah/Downloads/still_life_based_on_heathers_artwork.glb');
const fixtureName = basename(fixturePath);
const fixtureLength = statSync(fixturePath).size;
assert.ok(['js', 'shadow', 'rust'].includes(expectedImplementation),
  'HYPERSCOPE_EXPECT_ANIMCLIP must be js, shadow, or rust');
assert.ok(['implicit', 'js', 'shadow', 'rust'].includes(routeImplementation),
  'HYPERSCOPE_ANIMCLIP_ROUTE must be implicit, js, shadow, or rust');

const ownsTarget = targetId == null;
const fixtureServer = createServer((request, response) => {
  if (request.url !== '/animation-fixture.glb') {
    response.writeHead(404).end();
    return;
  }
  response.writeHead(200, {
    'Access-Control-Allow-Origin': '*',
    'Content-Length': fixtureLength,
    'Content-Type': 'model/gltf-binary',
  });
  createReadStream(fixturePath).pipe(response);
});
await new Promise((listen, reject) => {
  fixtureServer.once('error', reject);
  fixtureServer.listen(0, '127.0.0.1', listen);
});
const fixtureAddress = fixtureServer.address();
assert.ok(fixtureAddress && typeof fixtureAddress !== 'string');
const fixtureUrl = `http://127.0.0.1:${fixtureAddress.port}/animation-fixture.glb`;
const route = new URL(`http://127.0.0.1:${pagePort}/`);
route.searchParams.set('animate', '0');
route.searchParams.set('gfx', 'webgl2');
if (routeImplementation !== 'implicit') {
  route.searchParams.set('animclipimpl', routeImplementation);
}
const page = ownsTarget
  ? await (await fetch(
    `${cdpEndpoint}/json/new?${encodeURIComponent(route.href)}`,
    { method: 'PUT' },
  )).json()
  : (await (await fetch(`${cdpEndpoint}/json/list`)).json())
    .find(target => target.type === 'page' && target.id === targetId);
assert.ok(page, `animation-clip audit target ${targetId || '(new)'} was not found`);

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((open, reject) => {
  socket.addEventListener('open', open, { once: true });
  socket.addEventListener('error', reject, { once: true });
});

let nextCommandId = 1;
const pending = new Map();
socket.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id == null) return;
  const waiter = pending.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
  else waiter.resolve(message.result);
});

function command(method, params = {}) {
  const id = nextCommandId;
  nextCommandId += 1;
  return new Promise((resolveCommand, reject) => {
    pending.set(id, { resolve: resolveCommand, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression, { awaitPromise = true, returnByValue = true } = {}) {
  const response = await command('Runtime.evaluate', {
    expression,
    awaitPromise,
    returnByValue,
    userGesture: true,
  });
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.exception?.description
      || response.exceptionDetails.text);
  }
  return returnByValue ? response.result.value : response.result;
}

const snapshotSource = `(() => {
  const diagnostics = globalThis.__hyperscopeAnimationClipDiagnostics;
  const app = globalThis.__hyperscopeAppShadowController;
  const rustHost = document.getElementById('animation-clip-control-rust');
  const rustSelect = document.getElementById('animation-clip-select-rust');
  const browserHost = document.getElementById('animation-clip-control-browser');
  const browserSelect = document.getElementById('anim-sel');
  const selection = app?.animationRuntimeState?.()?.clipState;
  return {
    readyState: document.readyState,
    stats: document.getElementById('stats')?.textContent || '',
    diagnostics: diagnostics ? { ...diagnostics } : null,
    selection: selection ? {
      active: selection.active?.clip?.index ?? null,
      pending: selection.pending?.clip?.index ?? null,
      activeName: selection.active?.clip?.name ?? null,
      pendingName: selection.pending?.clip?.name ?? null,
    } : null,
    view: {
      rustHidden: rustHost?.hidden ?? null,
      browserHidden: browserHost?.hidden ?? null,
      rustOptions: rustSelect ? [...rustSelect.options].map(option => option.textContent) : [],
      browserOptions: browserSelect
        ? [...browserSelect.options]
          .filter(option => option.value !== '-1')
          .map(option => option.textContent)
        : [],
      rustValue: rustSelect?.value ?? null,
      browserValue: browserSelect?.value ?? null,
    },
    urlImplementation: new URLSearchParams(location.search).get('animclipimpl'),
    urlAnimation: new URLSearchParams(location.search).get('anim'),
  };
})()`;

async function snapshot() {
  return evaluate(snapshotSource);
}

async function waitFor(label, predicate, attempts = 600) {
  let current = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    current = await snapshot();
    if (predicate(current)) return current;
    await new Promise(resolveWait => setTimeout(resolveWait, 100));
  }
  throw new Error(`${label} did not settle: ${JSON.stringify(current)}`);
}

async function chooseClip(index) {
  await evaluate(`(() => {
    const implementation = ${JSON.stringify(expectedImplementation)};
    const select = document.getElementById(
      implementation === 'rust' ? 'animation-clip-select-rust' : 'anim-sel',
    );
    if (!select) throw new Error('animation selector is unavailable');
    select.value = ${JSON.stringify(String(index))};
    select.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
  })()`);
}

async function chooseClipsWithoutYield(indices) {
  await evaluate(`(() => {
    const implementation = ${JSON.stringify(expectedImplementation)};
    const select = document.getElementById(
      implementation === 'rust' ? 'animation-clip-select-rust' : 'anim-sel',
    );
    if (!select) throw new Error('animation selector is unavailable');
    for (const index of ${JSON.stringify(indices)}) {
      select.value = String(index);
      select.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
    }
  })()`);
}

async function installFixture() {
  const injectedName = await evaluate(`(async () => {
    const response = await fetch(${JSON.stringify(fixtureUrl)});
    if (!response.ok) throw new Error('animation fixture returned ' + response.status);
    const file = new File(
      [await response.arrayBuffer()],
      ${JSON.stringify(fixtureName)},
      { type: 'model/gltf-binary' },
    );
    const transfer = new DataTransfer();
    transfer.items.add(file);
    const accepted = document.getElementById('wrap').dispatchEvent(new DragEvent('drop', {
      bubbles: true,
      cancelable: true,
      dataTransfer: transfer,
    }));
    return { name: file.name, accepted };
  })()`);
  assert.equal(injectedName.name, fixtureName);
}

let audit;
try {
  await command('Runtime.enable');
  await command('DOM.enable');
  await waitFor('initial Hyperscope application', current =>
    current.readyState === 'complete' && current.diagnostics != null);
  await installFixture();

  const initial = await waitFor('two-clip fixture installation', current => {
    const options = expectedImplementation === 'rust'
      ? current.view.rustOptions : current.view.browserOptions;
    const expectedState = expectedImplementation === 'js' ? 'browser' : 'parity';
    return options.length === 2
      && options[0] === 'Action'
      && options[1] === 'EmptyAction'
      && current.view.browserValue === '0'
      && current.diagnostics?.implementation === expectedImplementation
      && current.diagnostics?.state === expectedState
      && current.selection?.activeName === 'Action'
      && (expectedImplementation === 'js'
        || (current.selection?.active === 0 && current.selection?.pending == null));
  });

  await chooseClip(1);
  const selected = await waitFor('clip 1 selection', current =>
    current.view.browserValue === '1'
      && current.diagnostics?.rendererResidentClip === 1
      && current.urlAnimation === '1'
      && (expectedImplementation === 'js'
        || (current.selection?.active === 1
          && current.selection?.pending == null
          && current.diagnostics?.state === 'parity')));

  await chooseClip(0);
  const restored = await waitFor('clip 0 restoration', current =>
    current.view.browserValue === '0'
      && current.diagnostics?.rendererResidentClip === 0
      && current.urlAnimation === initial.urlAnimation
      && (expectedImplementation === 'js'
        || (current.selection?.active === 0
          && current.selection?.pending == null
          && current.diagnostics?.state === 'parity')));

  let canceled = null;
  if (expectedImplementation !== 'js') {
    await chooseClipsWithoutYield([1, 0]);
    canceled = await waitFor('superseded renderer switch', current =>
      current.view.browserValue === '0'
        && current.diagnostics?.rendererResidentClip === 0
        && current.selection?.active === 0
        && current.selection?.pending == null
        && current.diagnostics?.state === 'parity'
        && current.diagnostics?.cancellationEffects
          > restored.diagnostics.cancellationEffects
        && current.diagnostics?.repairs > restored.diagnostics.repairs
        && current.diagnostics?.lastJobId == null);
  }

  let stale = null;
  if (expectedImplementation !== 'js') {
    stale = await evaluate(`(() => {
      const app = globalThis.__hyperscopeAppShadowController;
      const first = app.requestAnimationClip(1);
      if (!first.selection) throw new Error('stale oracle did not allocate a clip job');
      const canceled = app.requestAnimationClip(0);
      if (canceled.cancellations.length !== 1) {
        throw new Error('stale oracle did not cancel the pending clip job');
      }
      const completion = app.finishAnimationClipSelected(
        first.selection.job_id,
        first.selection.scene_request_id,
        first.selection.asset_id,
        first.selection.clip_index,
      );
      const state = app.animationRuntimeState().clipState;
      return {
        disposition: completion.commit.disposition,
        active: state.active?.clip?.index ?? null,
        pending: state.pending?.clip?.index ?? null,
      };
    })()`);
    assert.deepEqual(stale, {
      disposition: 'ignored_stale',
      active: 0,
      pending: null,
    });
  }

  const final = await snapshot();
  audit = {
    targetId: page.id,
    fixture: fixtureName,
    initial,
    selected,
    restored,
    canceled,
    stale,
    final,
  };

  assert.equal(final.diagnostics.mismatches, initial.diagnostics.mismatches);
  assert.equal(final.diagnostics.errors, initial.diagnostics.errors);
  if (expectedImplementation === 'rust') {
    assert.deepEqual(initial.view.rustOptions, ['Action', 'EmptyAction']);
    assert.equal(initial.view.rustHidden, false);
    assert.equal(initial.view.browserHidden, true);
    assert.ok(selected.diagnostics.authorityWrites > initial.diagnostics.authorityWrites);
    assert.ok(selected.diagnostics.completions > initial.diagnostics.completions);
    assert.equal(final.urlImplementation, null);
  } else {
    assert.equal(initial.view.rustHidden, true);
    assert.equal(initial.view.browserHidden, false);
  }
  if (expectedImplementation === 'shadow') {
    assert.ok(selected.diagnostics.completions > initial.diagnostics.completions);
    assert.equal(selected.diagnostics.authorityWrites, 0);
    assert.equal(final.urlImplementation, 'shadow');
  }
  if (expectedImplementation === 'js') {
    assert.equal(final.diagnostics.dispatches, 0);
    assert.equal(final.diagnostics.comparisons, 0);
    assert.equal(final.diagnostics.authorityWrites, 0);
    assert.equal(final.urlImplementation, 'js');
  }
  console.log(JSON.stringify(audit, null, 2));
} finally {
  socket.close();
  if (ownsTarget) {
    await fetch(`${cdpEndpoint}/json/close/${page.id}`);
  }
  await new Promise(close => fixtureServer.close(close));
}
