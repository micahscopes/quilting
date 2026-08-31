#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const requiredStyle = process.env.HYPERSCOPE_BACKEND_EVIDENCE_STYLE || 'normals';
const requireImageParity = process.env.HYPERSCOPE_REQUIRE_IMAGE_PARITY !== '0';
const focusPolicy = process.env.HYPERSCOPE_BACKEND_EVIDENCE_FOCUS || 'off';
const parityLimits = {
  coverageMismatchMillionths: Number(
    process.env.HYPERSCOPE_MAX_COVERAGE_MISMATCH_PPM || 5_000,
  ),
  rgbMeanAbsoluteErrorMillionths: Number(
    process.env.HYPERSCOPE_MAX_RGB_MEAN_ERROR_PPM || 5_000,
  ),
  rgbPixelsOver16Millionths: Number(
    process.env.HYPERSCOPE_MAX_RGB_OVER_16_PPM || 5_000,
  ),
};
const permittedStyles = new Set(['pbr', 'matcap', 'wire', 'normals', 'both', 'lod', 'stretch']);
assert.ok(permittedStyles.has(requiredStyle), `unsupported evidence style ${requiredStyle}`);
assert.ok(['off', 'preserve'].includes(focusPolicy),
  'HYPERSCOPE_BACKEND_EVIDENCE_FOCUS must be off or preserve');

const targetsBefore = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const originalPage = targetsBefore.find(target => target.type === 'page'
  && target.url.includes(`:${pagePort}/`));
const route = new URL(process.env.HYPERSCOPE_BACKEND_EVIDENCE_URL
  || `http://127.0.0.1:${pagePort}/`);
route.searchParams.set('_backend_evidence', String(Date.now()));
route.searchParams.set('gfx', 'webgpu');
route.searchParams.set('mode', requiredStyle);
route.searchParams.set('animate', '0');
if (focusPolicy === 'off') route.searchParams.set('fuzzy', '0');
route.searchParams.set('lodimpl', 'rust');

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
await command('Emulation.setDeviceMetricsOverride', {
  width: 960,
  height: 720,
  deviceScaleFactor: 1,
  mobile: false,
});

const snapshotExpression = `(async () => {
  const graphics = globalThis.__hyperscopeGraphicsBackend;
  const residency = await graphics?.refresh?.();
  return {
    url: location.href,
    effective: graphics?.effective ?? null,
    state: graphics?.state ?? null,
    mode: graphics?.renderMode ?? null,
    presentationArmed: Boolean(graphics?.presentationArmed),
    focusPostprocessRequested: Boolean(graphics?.focusPostprocessRequested),
    animationPlaying: document.getElementById('animate-toggle')
      ?.classList.contains('on') ?? null,
    animationTime: Number(document.getElementById('time')?.value ?? NaN),
    residency: residency ? {
      state: residency.state,
      presentationFrameAdmitted: Boolean(residency.presentationFrameAdmitted),
      presentationStyle: residency.presentationStyle ?? null,
      presentationColorFormat: residency.presentationColorFormat ?? null,
      presentationAlphaMode: residency.presentationAlphaMode ?? null,
      renderClearColor: residency.renderClearColor ?? null,
      presentationViewport: residency.presentationViewport ?? null,
      presentationFrames: Number(residency.presentationFrames || 0),
      framesSubmitted: Number(residency.framesSubmitted || 0),
      frameFailures: Number(residency.frameFailures || 0),
      lastFrameFailure: residency.lastFrameFailure ?? null,
      modelFaces: Number(residency.modelFaces || 0),
      sceneInstances: Number(residency.sceneInstances || 0),
      lastLogicalSubmission: residency.lastLogicalSubmission ?? null,
    } : null,
  };
})()`;

async function snapshot() {
  return evaluate(snapshotExpression);
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

async function waitForQuiescentPresentation(timeoutMs = 30_000, quietMs = 1_000) {
  const deadline = Date.now() + timeoutMs;
  let fingerprint = null;
  let unchangedSince = Date.now();
  let current = null;
  while (Date.now() < deadline) {
    current = await snapshot();
    const eligible = current.effective === 'webgpu'
      && current.state === 'presenting'
      && current.mode === requiredStyle
      && current.residency?.presentationFrameAdmitted
      && current.residency?.presentationStyle === requiredStyle
      && current.residency?.modelFaces > 0
      && current.residency?.sceneInstances > 0
      && current.animationPlaying === false;
    const nextFingerprint = eligible ? JSON.stringify([
      current.residency.framesSubmitted,
      current.residency.lastLogicalSubmission,
    ]) : null;
    if (nextFingerprint !== fingerprint) {
      fingerprint = nextFingerprint;
      unchangedSince = Date.now();
    } else if (eligible && Date.now() - unchangedSince >= quietMs) {
      return current;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`WebGPU presentation did not become quiescent: ${JSON.stringify(current)}`);
}

let audit;
try {
  await waitFor('WebGPU presentation', current =>
    current.effective === 'webgpu'
      && current.state === 'presenting'
      && current.mode === requiredStyle
      && current.residency?.presentationFrameAdmitted
      && current.residency?.presentationStyle === requiredStyle
      && current.residency?.modelFaces > 0
      && current.residency?.sceneInstances > 0
      && current.animationPlaying === false);
  const before = await waitForQuiescentPresentation();
  const requested = await evaluate(
    'globalThis.__hyperscopeGraphicsBackend.requestFrameEvidence()',
  );
  assert.equal(requested, true, 'backend rejected the frame-evidence request');
  const completed = await waitFor('offscreen parity frame', current =>
    current.residency?.framesSubmitted > before.residency.framesSubmitted);
  // The next supported frame renders both the incumbent WebGL2 oracle and the
  // WebGPU candidate before either readback is staged. Queue order makes one
  // comparison call sufficient once the submission counter advances.
  const report = await evaluate(
    'globalThis.__hyperscopeGraphicsBackend.compareFrameEvidence()',
  );
  const rgbAbsoluteError = report.image.absoluteChannelError
    .slice(0, 3)
    .reduce((sum, value) => sum + Number(value), 0);
  const rgbMeanAbsoluteErrorMillionths = Math.floor(
    rgbAbsoluteError * 1_000_000 / (report.image.comparedPixels * 3 * 255),
  );
  const rgbPixelsOver16Millionths = Math.floor(
    report.image.deltaProfile.rgbPixelsOver[5]
      * 1_000_000 / report.image.comparedPixels,
  );
  audit = {
    before,
    completed,
    report,
    parity: {
      rgbMeanAbsoluteErrorMillionths,
      rgbPixelsOver16Millionths,
      coverageMismatchMillionths: report.image.coverageMismatchMillionths,
    },
    parityLimits,
  };
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
assert.equal(audit.completed.residency.frameFailures, 0);
assert.ok(audit.report.workloadMismatch == null
    || Object.values(audit.report.workloadMismatch).every(value => value === false),
  'WebGL2 and WebGPU submitted different logical work');
assert.deepEqual(audit.report.viewport, audit.before.residency.presentationViewport);
assert.ok(audit.report.clearColor.every((value, index) =>
  Math.abs(value - [0.2, 0.2, 0.3, 1][index]) <= 1e-6),
  'backend evidence reported a noncanonical clear color');
assert.deepEqual(audit.before.residency.renderClearColor, audit.report.clearColor);
assert.equal(audit.before.residency.presentationAlphaMode, 'Opaque');
if (requireImageParity) {
  assert.ok(audit.parity.coverageMismatchMillionths
      <= parityLimits.coverageMismatchMillionths,
    `silhouette drift is ${audit.parity.coverageMismatchMillionths} ppm`);
  assert.ok(audit.parity.rgbMeanAbsoluteErrorMillionths
      <= parityLimits.rgbMeanAbsoluteErrorMillionths,
    `RGB mean absolute error is ${audit.parity.rgbMeanAbsoluteErrorMillionths} ppm`);
  assert.ok(audit.parity.rgbPixelsOver16Millionths
      <= parityLimits.rgbPixelsOver16Millionths,
    `RGB >16 drift is ${audit.parity.rgbPixelsOver16Millionths} ppm`);
}
