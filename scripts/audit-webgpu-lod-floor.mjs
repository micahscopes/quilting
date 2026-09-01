#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const targets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const originalPage = targets.find(target =>
  target.type === 'page' && target.url.includes(`:${pagePort}/`));
assert.ok(originalPage, 'a pre-existing Hyperscope page is required');

const route = new URL(process.env.HYPERSCOPE_LOD_FLOOR_URL
  || `http://127.0.0.1:${pagePort}/?glb=horse.glb`);
for (const [key, value] of [
  ['gfx', 'webgpu'],
  ['mode', 'wire'],
  ['minpx', '1'],
  ['animate', '0'],
  ['fuzzy', '0'],
  ['lodimpl', 'rust'],
]) route.searchParams.set(key, value);
route.searchParams.set('_lod_floor_evidence', String(Date.now()));

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

async function snapshot() {
  return evaluate(`(async () => {
    const graphics = globalThis.__hyperscopeGraphicsBackend;
    const residency = await graphics?.refresh?.();
    return {
      effective: graphics?.effective ?? null,
      state: graphics?.state ?? null,
      floor: Number(document.getElementById('min-px-sub')?.value ?? NaN),
      modelFaces: Number(residency?.modelFaces || 0),
      dispatches: Number(residency?.deviceLodDispatches || 0),
      failures: Number(residency?.frameFailures || 0),
      lastError: residency?.lastError ?? null,
    };
  })()`);
}

async function waitFor(label, predicate, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  let current = null;
  while (Date.now() < deadline) {
    current = await snapshot();
    if (predicate(current)) return current;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`${label} did not settle: ${JSON.stringify(current)}`);
}

async function residentWords() {
  return evaluate(`(async () => {
    const wasm = await import('./pkg/quilting_wasm.js');
    return Array.from(await wasm.mr_readWebGpuResidentLod());
  })()`);
}

function localEdgeCounts(packed) {
  const canonical = [
    1 << (packed & 15),
    1 << ((packed >>> 4) & 15),
    1 << ((packed >>> 8) & 15),
  ];
  switch ((packed >>> 12) & 7) {
    case 1: return [canonical[0], canonical[2], canonical[1]];
    case 2: return [canonical[1], canonical[0], canonical[2]];
    case 3: return [canonical[1], canonical[2], canonical[0]];
    case 4: return [canonical[2], canonical[0], canonical[1]];
    case 5: return [canonical[2], canonical[1], canonical[0]];
    default: return canonical;
  }
}

function summarize(floor, words) {
  const exponentDistribution = {};
  let visibleFaces = 0;
  let edgeSubdivisionSum = 0;
  let maxExponent = 0;
  for (const packed of words) {
    const exponents = [packed & 15, (packed >>> 4) & 15, (packed >>> 8) & 15];
    const key = exponents.join('/');
    exponentDistribution[key] = (exponentDistribution[key] || 0) + 1;
    visibleFaces += (packed & (1 << 15)) !== 0 ? 1 : 0;
    maxExponent = Math.max(maxExponent, ...exponents);
    edgeSubdivisionSum += localEdgeCounts(packed).reduce((sum, value) => sum + value, 0);
  }
  return {
    floor,
    faces: words.length,
    visibleFaces,
    maxExponent,
    edgeSubdivisionSum,
    exponentDistribution,
  };
}

async function setFloor(floor, previousDispatches) {
  await evaluate(`(() => {
    const input = document.getElementById('min-px-sub');
    input.value = ${JSON.stringify(String(floor))};
    input.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  return waitFor(`LOD floor ${floor}`, current =>
    current.effective === 'webgpu'
      && current.state === 'presenting'
      && current.floor === floor
      && current.dispatches > previousDispatches
      && current.failures === 0);
}

try {
  await command('Runtime.enable');
  await command('Log.enable');
  const initial = await waitFor('initial WebGPU LOD epoch', current =>
    current.effective === 'webgpu'
      && current.state === 'presenting'
      && current.floor === 1
      && current.modelFaces > 0
      && current.dispatches > 0);
  const floors = [{ state: initial, words: await residentWords() }];
  for (const floor of [16, 64]) {
    const state = await setFloor(floor, floors.at(-1).state.dispatches);
    floors.push({ state, words: await residentWords() });
  }

  for (let index = 1; index < floors.length; index += 1) {
    const finer = floors[index - 1].words;
    const coarser = floors[index].words;
    assert.equal(coarser.length, finer.length);
    for (let face = 0; face < finer.length; face += 1) {
      const finerEdges = localEdgeCounts(finer[face]);
      const coarserEdges = localEdgeCounts(coarser[face]);
      for (let edge = 0; edge < 3; edge += 1) {
        assert.ok(coarserEdges[edge] <= finerEdges[edge],
          `face ${face} edge ${edge} grew from ${finerEdges[edge]} to ${coarserEdges[edge]}`);
      }
    }
  }

  const audit = {
    route: route.href,
    floors: floors.map(({ state, words }) => ({
      state,
      resident: summarize(state.floor, words),
    })),
    runtimeErrors,
  };
  console.log(JSON.stringify(audit, null, 2));
  assert.deepEqual(runtimeErrors, []);
  assert.ok(audit.floors[1].resident.edgeSubdivisionSum
    < audit.floors[0].resident.edgeSubdivisionSum);
  assert.ok(audit.floors[2].resident.edgeSubdivisionSum
    <= audit.floors[1].resident.edgeSubdivisionSum);
  assert.equal(audit.floors.at(-1).state.lastError, null);
} finally {
  await activate(originalPage.id);
  socket.close();
  await fetch(`${cdpEndpoint}/json/close/${page.id}`);
}
