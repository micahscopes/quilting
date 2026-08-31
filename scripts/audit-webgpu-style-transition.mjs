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

function auditWebGpuStyleTransition() {
  const graphics = globalThis.__hyperscopeGraphicsBackend;
  const policy = globalThis.__hyperscopeGraphicsPresentationPolicy;
  if (!graphics || !policy) throw new Error('graphics diagnostics are unavailable');
  const buttonFor = mode => document.querySelector(
    `#render-btns button[data-v="${mode}"]`,
  );
  const snapshot = async () => {
    const residency = await graphics.refresh();
    return {
      mode: graphics.renderMode,
      armed: graphics.presentationArmed,
      focusPostprocessRequested: graphics.focusPostprocessRequested,
      effective: graphics.effective,
      state: graphics.state,
      policy: policy.lastEffective,
      residency: {
        state: residency?.state,
        style: residency?.presentationStyle,
        admitted: residency?.presentationFrameAdmitted,
        frames: Number(residency?.presentationFrames || 0),
        deviceLodDispatches: Number(residency?.deviceLodDispatches || 0),
        frameFailures: Number(residency?.frameFailures || 0),
        lastFrameFailure: residency?.lastFrameFailure ?? null,
      },
    };
  };
  const waitFor = async predicate => {
    let current = null;
    for (let attempt = 0; attempt < 50; attempt += 1) {
      current = await snapshot();
      if (predicate(current)) return current;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return current;
  };

  return (async () => {
    const before = await waitFor(current => current.effective === 'webgpu');
    const originalMode = before.mode;
    const targetMode = originalMode === 'matcap' ? 'normals' : 'matcap';
    const originalButton = buttonFor(originalMode);
    const targetButton = buttonFor(targetMode);
    if (!originalButton || !targetButton) {
      throw new Error(`cannot audit ${originalMode} -> ${targetMode}`);
    }
    let transitioned;
    try {
      targetButton.click();
      await new Promise(resolve => setTimeout(resolve, 500));
      transitioned = await waitFor(current => current.mode === targetMode
        && (before.focusPostprocessRequested
          ? current.state === 'unsupported-mode'
          : current.effective === 'webgpu'
            && current.residency.style === targetMode));
    } finally {
      originalButton.click();
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    const restored = await waitFor(current => current.mode === originalMode
      && current.effective === 'webgpu'
      && current.residency.style === originalMode);

    let withoutFocus = null;
    if (before.focusPostprocessRequested) {
      const focusModeButtons = [...document.querySelectorAll('#fuzzy-mode-btns button')];
      const originalFocusMode = focusModeButtons.find(button => button.classList.contains('a'));
      const neutralFocusMode = focusModeButtons.find(button => button.dataset.v === '0');
      const focusToggle = document.getElementById('fuzzy-toggle');
      const originalToggleOn = focusToggle?.classList.contains('on') === true;
      if (!originalFocusMode || !neutralFocusMode || !focusToggle) {
        throw new Error('focus controls are unavailable for composition audit');
      }
      try {
        if (originalFocusMode !== neutralFocusMode) neutralFocusMode.click();
        if (focusToggle.classList.contains('on')) focusToggle.click();
        await waitFor(current => !current.focusPostprocessRequested
          && current.effective === 'webgpu');
        targetButton.click();
        await new Promise(resolve => setTimeout(resolve, 500));
        withoutFocus = await waitFor(current => current.mode === targetMode
          && current.effective === 'webgpu'
          && current.residency.style === targetMode);
      } finally {
        originalButton.click();
        if (originalToggleOn !== focusToggle.classList.contains('on')) focusToggle.click();
        if (originalFocusMode !== neutralFocusMode) originalFocusMode.click();
      }
      await waitFor(current => current.mode === originalMode
        && current.focusPostprocessRequested
        && current.effective === 'webgpu');
    }
    return { before, transitioned, withoutFocus, restored };
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
      expression: `(${auditWebGpuStyleTransition.toString()})()`,
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
if (audit.before.focusPostprocessRequested) {
  assert.equal(audit.transitioned.effective, 'webgl2');
  assert.equal(audit.transitioned.state, 'unsupported-mode');
  assert.equal(audit.transitioned.policy.supportsRequestedStyle, false);
  assert.equal(audit.withoutFocus.effective, 'webgpu');
  assert.equal(audit.withoutFocus.residency.style, audit.withoutFocus.mode);
} else {
  assert.equal(audit.transitioned.effective, 'webgpu');
  assert.equal(audit.transitioned.residency.style, audit.transitioned.mode);
}
assert.equal(audit.restored.effective, 'webgpu');
assert.equal(audit.restored.residency.style, audit.restored.mode);
