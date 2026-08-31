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

function auditGraphicsPresentationPolicy() {
  const graphics = globalThis.__hyperscopeGraphicsBackend;
  const policy = globalThis.__hyperscopeGraphicsPresentationPolicy;
  if (!graphics || !policy) {
    throw new Error('graphics presentation diagnostics are unavailable');
  }
  const buttonFor = mode => document.querySelector(
    `#render-btns button[data-v="${mode}"]`,
  );
  const snapshot = async () => {
    await graphics.refresh();
    return {
      mode: graphics.renderMode,
      effective: graphics.effective,
      state: graphics.state,
      comparisons: Number(policy.comparisons || 0),
      mismatchCount: Number(policy.mismatchCount || 0),
      errors: Number(policy.errors || 0),
      implementation: policy.implementation,
      authority: policy.authority,
      browser: policy.lastBrowser,
      rust: policy.lastRust,
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
      'initial policy comparison',
      current => current.effective === 'webgpu'
        && current.browser?.presentWebgpu === true
        && current.rust?.presentWebgpu === true,
    );
    const originalMode = before.mode;
    const originalButton = buttonFor(originalMode);
    const unsupportedButton = buttonFor('fz-weight');
    if (!originalButton || !unsupportedButton) {
      throw new Error(`cannot audit policy from render mode ${originalMode}`);
    }

    let unsupported;
    try {
      unsupportedButton.click();
      unsupported = await waitFor(
        'unsupported policy',
        current => current.mode === 'fz-weight'
          && current.effective === 'webgl2'
          && current.browser?.phase === 'unsupported-mode'
          && current.rust?.phase === 'unsupported-mode',
      );
    } finally {
      originalButton.click();
    }
    const recovered = await waitFor(
      'restored policy',
      current => current.mode === originalMode
        && current.effective === 'webgpu'
        && current.browser?.phase === 'presenting'
        && current.rust?.phase === 'presenting',
    );
    return { before, unsupported, recovered };
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
      expression: `(${auditGraphicsPresentationPolicy.toString()})()`,
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
assert.equal(audit.recovered.mismatchCount, audit.before.mismatchCount);
assert.equal(audit.recovered.errors, audit.before.errors);
assert.equal(audit.before.implementation, 'rust');
assert.equal(audit.recovered.authority, 'hyperscope-app');
assert.deepEqual(audit.unsupported.browser, audit.unsupported.rust);
assert.deepEqual(audit.recovered.browser, audit.recovered.rust);
