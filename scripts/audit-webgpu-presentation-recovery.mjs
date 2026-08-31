#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const targetId = process.env.HYPERSCOPE_TARGET_ID || null;

const targets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const page = targets.find(target =>
  target.type === 'page' && target.url.includes(`:${pagePort}/`)
    && (!targetId || target.id === targetId),
);
assert.ok(page, `no Hyperscope page found on port ${pagePort}`);

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});

function auditPresentationRecovery() {
  const graphics = globalThis.__hyperscopeGraphicsBackend;
  const lod = globalThis.__hyperscopeWebGpuLodAuthority;
  if (!graphics || !lod) throw new Error('WebGPU diagnostics are unavailable');

  const buttonFor = mode => document.querySelector(
    `#render-btns button[data-v="${mode}"]`,
  );
  const focusDiagnosticButtonFor = value => document.querySelector(
    `#fuzzy-debug-btns button[data-v="${value}"]`,
  );
  const snapshot = async () => {
    const residency = await graphics.refresh();
    return {
      effective: graphics.effective,
      state: graphics.state,
      mode: graphics.renderMode,
      admitted: residency?.presentationFrameAdmitted === true,
      presentationFrames: Number(residency?.presentationFrames || 0),
      deviceLodDispatches: Number(residency?.deviceLodDispatches || 0),
      frameFailures: Number(residency?.frameFailures || 0),
      lastFrameFailure: residency?.lastFrameFailure ?? null,
      lodActive: Boolean(lod.effectiveActive),
      lodState: lod.state,
    };
  };
  const waitFor = async (label, predicate) => {
    let current = null;
    for (let attempt = 0; attempt < 50; attempt += 1) {
      current = await snapshot();
      if (predicate(current)) return current;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error(`${label} did not settle: ${JSON.stringify(current)}`);
  };

  return (async () => {
    const before = await waitFor(
      'initial WebGPU presentation',
      current => current.effective === 'webgpu'
        && current.admitted
        && current.lodActive,
    );
    const originalMode = before.mode;
    const originalButton = buttonFor(originalMode);
    const fallbackButton = focusDiagnosticButtonFor('1');
    const compositeButton = focusDiagnosticButtonFor('0');
    if (!originalButton || !fallbackButton || !compositeButton
        || originalMode === 'fz-weight') {
      throw new Error(`cannot audit recovery from render mode ${originalMode}`);
    }

    let fallback;
    try {
      fallbackButton.click();
      fallback = await waitFor(
        'unsupported-mode fallback',
        current => current.mode === 'fz-weight'
          && current.effective === 'webgl2'
          && !current.lodActive,
      );
    } finally {
      compositeButton.click();
      originalButton.click();
    }
    const recovered = await waitFor(
      'WebGPU presentation recovery',
      current => current.mode === originalMode
        && current.effective === 'webgpu'
        && current.admitted
        && current.lodActive,
    );
    return { before, fallback, recovered };
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
      expression: `(${auditPresentationRecovery.toString()})()`,
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
assert.equal(audit.before.effective, 'webgpu');
assert.equal(audit.fallback.effective, 'webgl2');
assert.equal(audit.fallback.lodActive, false);
assert.equal(audit.recovered.effective, 'webgpu');
assert.equal(audit.recovered.admitted, true);
assert.equal(audit.recovered.lodActive, true);
assert.ok(
  audit.recovered.presentationFrames > audit.before.presentationFrames,
  'recovery did not present a fresh WebGPU frame',
);
assert.ok(
  audit.recovered.deviceLodDispatches > audit.before.deviceLodDispatches,
  'recovery did not classify a fresh complete device epoch',
);
