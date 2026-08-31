#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const targetId = process.env.HYPERSCOPE_TARGET_ID || null;
const expectedImplementation = process.env.HYPERSCOPE_EXPECT_RENDERSTATE || 'rust';
assert.ok(['js', 'rust'].includes(expectedImplementation),
  'HYPERSCOPE_EXPECT_RENDERSTATE must be js or rust');
const targets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const page = targets.find(target => target.type === 'page'
  && target.url.includes(`:${pagePort}/`)
  && (targetId ? target.id === targetId
    : expectedImplementation === 'js'
      ? target.url.includes('renderstateimpl=js')
      : !target.url.includes('renderstateimpl=js')));
assert.ok(page, `no ${expectedImplementation} render-settings Hyperscope page found on port ${pagePort}`);

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});

function auditRustRenderSettings() {
  const waitFor = async (label, predicate) => {
    let current = null;
    for (let attempt = 0; attempt < 80; attempt += 1) {
      current = snapshot();
      if (predicate(current)) return current;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error(`${label} did not settle: ${JSON.stringify(current)}`);
  };
  const snapshot = () => {
    const diagnostics = globalThis.__hyperscopeRenderSettings;
    const app = globalThis.__hyperscopeAppShadowController;
    const renderView = document.getElementById('render-controls-rust-view');
    const focusView = document.getElementById('focus-postprocess-controls-rust-view');
    const renderInputs = [...(renderView?.querySelectorAll('input') || [])];
    const focusInputs = [...(focusView?.querySelectorAll('input') || [])];
    const params = new URLSearchParams(location.search);
    return {
      diagnostics: diagnostics ? {
        implementation: diagnostics.implementation,
        state: diagnostics.state,
        comparisons: Number(diagnostics.comparisons || 0),
        mismatches: Number(diagnostics.mismatches || 0),
        errors: Number(diagnostics.errors || 0),
        authorityRollbacks: Number(diagnostics.authorityRollbacks || 0),
        lastDispatchSource: diagnostics.lastDispatchSource ?? null,
        viewAuthority: diagnostics.viewAuthority,
        viewMountAttempts: Number(diagnostics.viewMountAttempts || 0),
        viewMountErrors: Number(diagnostics.viewMountErrors || 0),
      } : null,
      render: app?.snapshot?.().renderSettings ?? null,
      graphicsRenderMode: globalThis.__hyperscopeGraphicsBackend?.renderMode ?? null,
      url: {
        renderstateimpl: params.get('renderstateimpl'),
        density: params.get('density'),
        fuzzyRadius: params.get('fradius'),
        focusDiagnostic: params.get('fdebug'),
      },
      view: {
        renderRustHidden: document.getElementById('render-controls-rust')?.hidden,
        renderBrowserHidden: document.getElementById('render-settings-browser')?.hidden,
        modeBrowserHidden: document.getElementById('render-mode-browser')?.hidden,
        focusRustHidden: document.getElementById('focus-postprocess-controls-rust')?.hidden,
        focusBrowserHidden: document.getElementById('focus-postprocess-controls-browser')?.hidden,
        renderInputCount: renderInputs.length,
        focusInputCount: focusInputs.length,
      },
    };
  };
  const setInput = (selector, index, value) => {
    const inputs = [...document.querySelectorAll(`${selector} input`)];
    const input = inputs[index];
    if (!input) throw new Error(`missing ${selector} input ${index}`);
    input.value = String(value);
    input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
  };

  return (async () => {
    const before = await waitFor(
      'Rust render-settings authority',
      current => current.diagnostics?.state === 'authoritative'
        && current.diagnostics.viewAuthority === 'hyperscope-web'
        && current.render != null,
    );
    const originalDensity = before.render.density;
    const originalFocusRadius = before.render.focusPostprocess.blurRadiusPixels;
    const originalFocusDiagnostic = before.render.focusPostprocess.diagnosticView;
    const nextDensity = originalDensity === 137 ? 138 : 137;
    const nextFocusRadius = originalFocusRadius === 57 ? 58 : 57;
    let density;
    let focus;
    let diagnostic;
    try {
      setInput('#render-controls-rust-view', 1, nextDensity);
      density = await waitFor(
        'Rust density control',
        current => current.render?.density === nextDensity
          && current.url.density === String(nextDensity),
      );
      setInput('#focus-postprocess-controls-rust-view', 1, nextFocusRadius);
      focus = await waitFor(
        'Rust focus-radius control',
        current => current.render?.focusPostprocess?.blurRadiusPixels === nextFocusRadius
          && current.url.fuzzyRadius === String(nextFocusRadius),
      );
      const weight = [...document.querySelectorAll(
        '#focus-postprocess-controls-rust-view button',
      )].find(button => button.textContent.trim() === 'Weight');
      if (!weight) throw new Error('Rust focus Weight diagnostic control is unavailable');
      weight.click();
      diagnostic = await waitFor(
        'Rust focus diagnostic control',
        current => current.render?.focusPostprocess?.diagnosticView === 1
          && current.render.style === before.render.style
          && current.graphicsRenderMode === 'fz-weight'
          && current.url.focusDiagnostic === '1',
      );
    } finally {
      setInput('#render-controls-rust-view', 1, originalDensity);
      setInput('#focus-postprocess-controls-rust-view', 1, originalFocusRadius);
      const composite = [...document.querySelectorAll(
        '#focus-postprocess-controls-rust-view button',
      )].find(button => button.textContent.trim() === 'Composite');
      if (originalFocusDiagnostic === 0) composite?.click();
    }
    const restored = await waitFor(
      'restored render settings',
      current => current.render?.density === originalDensity
        && current.render?.focusPostprocess?.blurRadiusPixels === originalFocusRadius
        && current.render?.focusPostprocess?.diagnosticView === originalFocusDiagnostic
        && current.graphicsRenderMode === before.graphicsRenderMode
        && current.url.density === before.url.density
        && current.url.fuzzyRadius === before.url.fuzzyRadius
        && current.url.focusDiagnostic === before.url.focusDiagnostic,
    );
    return { before, density, focus, diagnostic, restored };
  })();
}

function auditJavascriptRenderSettingsRollback() {
  const snapshot = () => {
    const diagnostics = globalThis.__hyperscopeRenderSettings;
    const params = new URLSearchParams(location.search);
    return {
      diagnostics: diagnostics ? {
        implementation: diagnostics.implementation,
        state: diagnostics.state,
        dispatches: Number(diagnostics.dispatches || 0),
        comparisons: Number(diagnostics.comparisons || 0),
        mismatches: Number(diagnostics.mismatches || 0),
        errors: Number(diagnostics.errors || 0),
        viewAuthority: diagnostics.viewAuthority,
        viewMountAttempts: Number(diagnostics.viewMountAttempts || 0),
      } : null,
      url: {
        renderstateimpl: params.get('renderstateimpl'),
        density: params.get('density'),
        focusDiagnostic: params.get('fdebug'),
      },
      view: {
        renderRustHidden: document.getElementById('render-controls-rust')?.hidden,
        renderBrowserHidden: document.getElementById('render-settings-browser')?.hidden,
        modeBrowserHidden: document.getElementById('render-mode-browser')?.hidden,
        focusRustHidden: document.getElementById('focus-postprocess-controls-rust')?.hidden,
        focusBrowserHidden: document.getElementById('focus-postprocess-controls-browser')?.hidden,
      },
      density: Number(document.getElementById('tess-density')?.value),
      graphicsRenderMode: globalThis.__hyperscopeGraphicsBackend?.renderMode ?? null,
    };
  };
  const waitFor = async (label, predicate) => {
    let current = null;
    for (let attempt = 0; attempt < 80; attempt += 1) {
      current = snapshot();
      if (predicate(current)) return current;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error(`${label} did not settle: ${JSON.stringify(current)}`);
  };

  return (async () => {
    const before = await waitFor(
      'JavaScript render-settings rollback',
      current => current.diagnostics?.state === 'disabled'
        && current.diagnostics.viewAuthority === 'browser-fallback',
    );
    const input = document.getElementById('tess-density');
    if (!input) throw new Error('browser tessellation-density control is unavailable');
    const originalDensity = Number(input.value);
    const nextDensity = originalDensity === 137 ? 138 : 137;
    try {
      input.value = String(nextDensity);
      input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
      const changed = await waitFor(
        'JavaScript density control',
        current => current.density === nextDensity
          && current.url.density === String(nextDensity),
      );
      const weight = document.querySelector('#fuzzy-debug-btns button[data-v="1"]');
      if (!weight) throw new Error('browser focus Weight diagnostic control is unavailable');
      weight.click();
      const diagnostic = await waitFor(
        'JavaScript focus diagnostic control',
        current => current.graphicsRenderMode === 'fz-weight'
          && current.url.focusDiagnostic === '1',
      );
      return { before, changed, diagnostic };
    } finally {
      input.value = String(originalDensity);
      input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
      document.querySelector('#fuzzy-debug-btns button[data-v="0"]')?.click();
      await waitFor(
        'restored JavaScript density control',
        current => current.density === originalDensity
          && current.url.density === before.url.density
          && current.graphicsRenderMode === before.graphicsRenderMode
          && current.url.focusDiagnostic === before.url.focusDiagnostic,
      );
    }
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
      expression: expectedImplementation === 'rust'
        ? `(${auditRustRenderSettings.toString()})()`
        : `(${auditJavascriptRenderSettingsRollback.toString()})()`,
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
if (expectedImplementation === 'rust') {
  assert.equal(audit.before.diagnostics.implementation, 'rust');
  assert.equal(audit.before.diagnostics.mismatches, audit.restored.diagnostics.mismatches);
  assert.equal(audit.before.diagnostics.errors, audit.restored.diagnostics.errors);
  assert.equal(audit.before.diagnostics.authorityRollbacks,
    audit.restored.diagnostics.authorityRollbacks);
  assert.equal(audit.restored.diagnostics.viewMountErrors, 0);
  assert.equal(audit.focus.diagnostics.lastDispatchSource, 'leptos-control');
  assert.equal(audit.diagnostic.render.style, audit.before.render.style);
  assert.equal(audit.diagnostic.render.focusPostprocess.diagnosticView, 1);
  assert.deepEqual(audit.before.view, {
    renderRustHidden: false,
    renderBrowserHidden: true,
    modeBrowserHidden: true,
    focusRustHidden: false,
    focusBrowserHidden: true,
    renderInputCount: 5,
    focusInputCount: 9,
  });
} else {
  assert.equal(audit.before.diagnostics.implementation, 'js');
  assert.deepEqual(audit.before.view, {
    renderRustHidden: true,
    renderBrowserHidden: false,
    modeBrowserHidden: false,
    focusRustHidden: true,
    focusBrowserHidden: false,
  });
  assert.equal(audit.changed.diagnostics.dispatches, 0);
  assert.equal(audit.changed.diagnostics.comparisons, 0);
  assert.equal(audit.changed.diagnostics.mismatches, 0);
  assert.equal(audit.changed.diagnostics.errors, 0);
  assert.equal(audit.changed.diagnostics.viewMountAttempts, 0);
  assert.equal(audit.diagnostic.graphicsRenderMode, 'fz-weight');
}
