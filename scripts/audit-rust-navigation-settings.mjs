#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const targetId = process.env.HYPERSCOPE_TARGET_ID || null;
const expectedImplementation = process.env.HYPERSCOPE_EXPECT_NAVSTATE || 'rust';
assert.ok(['js', 'shadow', 'rust'].includes(expectedImplementation),
  'HYPERSCOPE_EXPECT_NAVSTATE must be js, shadow, or rust');

const targets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const page = targets.find(target => target.type === 'page'
  && target.url.includes(`:${pagePort}/`)
  && (!targetId || target.id === targetId));
assert.ok(page, `no ${expectedImplementation} navigation-settings page found on port ${pagePort}`);

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});

function auditNavigationSettings(expected) {
  const snapshot = () => {
    const diagnostics = globalThis.__hyperscopeNavigationSettings;
    const app = globalThis.__hyperscopeAppShadowController;
    const rustView = document.getElementById('navigation-controls-rust-view');
    const params = new URLSearchParams(location.search);
    const browserValues = [
      'focus-transition',
      'walk-smoothing',
      'walk-align',
      'walk-speed',
      'walk-scale',
      'walk-height',
    ].map(id => Number(document.getElementById(id)?.value));
    return {
      diagnostics: diagnostics ? {
        implementation: diagnostics.implementation,
        state: diagnostics.state,
        dispatches: Number(diagnostics.dispatches || 0),
        comparisons: Number(diagnostics.comparisons || 0),
        mismatches: Number(diagnostics.mismatches || 0),
        authorityWrites: Number(diagnostics.authorityWrites || 0),
        authorityRollbacks: Number(diagnostics.authorityRollbacks || 0),
        errors: Number(diagnostics.errors || 0),
        lastDispatchSource: diagnostics.lastDispatchSource ?? null,
        lastBrowser: diagnostics.lastBrowser ?? null,
        lastRust: diagnostics.lastRust ?? null,
        viewAuthority: diagnostics.viewAuthority,
        viewMountAttempts: Number(diagnostics.viewMountAttempts || 0),
        viewMountErrors: Number(diagnostics.viewMountErrors || 0),
      } : null,
      navigation: app?.snapshot?.().navigationSettings ?? null,
      url: {
        implementation: params.get('navstateimpl'),
        transition: params.get('interp'),
        smoothing: params.get('walksmooth'),
        tangentPull: params.get('walkalign'),
        speed: params.get('walkspeed'),
        scale: params.get('walkscale'),
        eyeHeight: params.get('walkheight'),
      },
      view: {
        rustHidden: document.getElementById('navigation-controls-rust')?.hidden,
        browserHidden: document.getElementById('navigation-controls-browser')?.hidden,
        rustInputCount: rustView?.querySelectorAll('input').length || 0,
      },
      browserValues,
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
  const setInput = (selector, index, value) => {
    const inputs = [...document.querySelectorAll(`${selector} input`)];
    const input = inputs[index];
    if (!input) throw new Error(`missing ${selector} input ${index}`);
    input.value = String(value);
    input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
  };
  const setBrowserInput = (id, value) => {
    const input = document.getElementById(id);
    if (!input) throw new Error(`missing browser input ${id}`);
    input.value = String(value);
    input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
  };

  return (async () => {
    const targetState = expected === 'rust'
      ? 'authoritative'
      : expected === 'shadow' ? 'observing' : 'disabled';
    const before = await waitFor(
      `${expected} navigation-settings boundary`,
      current => current.diagnostics?.implementation === expected
        && current.diagnostics.state === targetState
        && (expected === 'js' || current.navigation != null)
        && (expected !== 'rust'
          || current.diagnostics.viewAuthority === 'hyperscope-web'),
    );
    const original = before.navigation;
    const originalBrowser = before.browserValues;
    const nextTransitionSeconds = original?.transitionSeconds === 1.23 ? 1.24 : 1.23;
    const nextTransitionCentiseconds = Math.round(nextTransitionSeconds * 100);
    const nextSmoothingSeconds = original?.smoothingSeconds === 0.42 ? 0.43 : 0.42;
    const nextSmoothingCentiseconds = Math.round(nextSmoothingSeconds * 100);
    const nextTangentPullFraction = original?.tangentPullFraction === 0.36 ? 0.37 : 0.36;
    const nextTangentPullPercent = Math.round(nextTangentPullFraction * 100);
    const nextSpeed = original?.speedOctaveSteps === 37 ? 38 : 37;
    const nextScale = original?.bodyScaleOctaveSteps === -41 ? -42 : -41;
    const nextEyeHeight = original?.eyeHeightOctaveSteps === 29 ? 30 : 29;
    let transition;
    let speed;
    let complete;
    try {
      if (expected === 'rust') {
        setInput('#navigation-controls-rust-view', 0, nextTransitionSeconds);
      } else {
        setBrowserInput('focus-transition', nextTransitionCentiseconds);
      }
      transition = await waitFor(
        `${expected} transition edit`,
        current => current.url.transition === String(nextTransitionCentiseconds)
          && (expected === 'js'
            || current.navigation?.transitionSeconds === nextTransitionSeconds),
      );

      if (expected === 'rust') {
        setInput('#navigation-controls-rust-view', 3, nextSpeed);
      } else {
        setBrowserInput('walk-speed', nextSpeed);
      }
      speed = await waitFor(
        `${expected} walk-speed edit`,
        current => current.url.speed === String(nextSpeed)
          && (expected === 'js'
            || current.navigation?.speedOctaveSteps === nextSpeed),
      );
      if (expected === 'rust') {
        setInput('#navigation-controls-rust-view', 1, nextSmoothingSeconds);
        setInput('#navigation-controls-rust-view', 2, nextTangentPullFraction);
        setInput('#navigation-controls-rust-view', 4, nextScale);
        setInput('#navigation-controls-rust-view', 5, nextEyeHeight);
      } else {
        setBrowserInput('walk-smoothing', nextSmoothingCentiseconds);
        setBrowserInput('walk-align', nextTangentPullPercent);
        setBrowserInput('walk-scale', nextScale);
        setBrowserInput('walk-height', nextEyeHeight);
      }
      complete = await waitFor(
        `${expected} complete navigation-settings edit`,
        current => current.url.smoothing === String(nextSmoothingCentiseconds)
          && current.url.tangentPull === String(nextTangentPullPercent)
          && current.url.scale === String(nextScale)
          && current.url.eyeHeight === String(nextEyeHeight)
          && (expected === 'js'
            || (current.navigation?.smoothingSeconds === nextSmoothingSeconds
              && current.navigation?.tangentPullFraction === nextTangentPullFraction
              && current.navigation?.bodyScaleOctaveSteps === nextScale
              && current.navigation?.eyeHeightOctaveSteps === nextEyeHeight)),
      );
    } finally {
      if (expected === 'rust') {
        setInput('#navigation-controls-rust-view', 0, original.transitionSeconds);
        setInput('#navigation-controls-rust-view', 1, original.smoothingSeconds);
        setInput('#navigation-controls-rust-view', 2, original.tangentPullFraction);
        setInput('#navigation-controls-rust-view', 3, original.speedOctaveSteps);
        setInput('#navigation-controls-rust-view', 4, original.bodyScaleOctaveSteps);
        setInput('#navigation-controls-rust-view', 5, original.eyeHeightOctaveSteps);
      } else {
        setBrowserInput('focus-transition', originalBrowser[0]);
        setBrowserInput('walk-smoothing', originalBrowser[1]);
        setBrowserInput('walk-align', originalBrowser[2]);
        setBrowserInput('walk-speed', originalBrowser[3]);
        setBrowserInput('walk-scale', originalBrowser[4]);
        setBrowserInput('walk-height', originalBrowser[5]);
      }
    }
    const restored = await waitFor(
      `${expected} navigation-settings restoration`,
      current => current.url.transition === before.url.transition
        && current.url.smoothing === before.url.smoothing
        && current.url.tangentPull === before.url.tangentPull
        && current.url.speed === before.url.speed
        && current.url.scale === before.url.scale
        && current.url.eyeHeight === before.url.eyeHeight
        && current.browserValues.every((value, index) => value === originalBrowser[index])
        && (expected === 'js'
          || (current.navigation?.transitionSeconds === original.transitionSeconds
            && current.navigation?.smoothingSeconds === original.smoothingSeconds
            && current.navigation?.tangentPullFraction === original.tangentPullFraction
            && current.navigation?.speedOctaveSteps === original.speedOctaveSteps
            && current.navigation?.bodyScaleOctaveSteps === original.bodyScaleOctaveSteps
            && current.navigation?.eyeHeightOctaveSteps === original.eyeHeightOctaveSteps)),
    );
    return { before, transition, speed, complete, restored };
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
      expression: `(${auditNavigationSettings.toString()})(${JSON.stringify(expectedImplementation)})`,
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

assert.equal(audit.before.diagnostics.implementation, expectedImplementation);
assert.equal(audit.before.diagnostics.mismatches, audit.restored.diagnostics.mismatches);
assert.equal(audit.before.diagnostics.errors, audit.restored.diagnostics.errors);
assert.equal(audit.before.diagnostics.authorityRollbacks,
  audit.restored.diagnostics.authorityRollbacks);
if (expectedImplementation === 'rust') {
  assert.equal(audit.before.diagnostics.viewAuthority, 'hyperscope-web');
  assert.equal(audit.restored.diagnostics.viewMountErrors, 0);
  assert.equal(audit.complete.diagnostics.lastDispatchSource, 'leptos-control');
  assert.deepEqual(audit.before.view, {
    rustHidden: false,
    browserHidden: true,
    rustInputCount: 6,
  });
} else if (expectedImplementation === 'shadow') {
  assert.equal(audit.before.diagnostics.state, 'observing');
  assert.ok(audit.complete.diagnostics.comparisons > audit.before.diagnostics.comparisons);
  assert.equal(audit.complete.diagnostics.lastBrowser.speedOctaveSteps, 37);
  const { revision: _revision, ...rustSettings } = audit.complete.diagnostics.lastRust;
  assert.deepEqual(audit.complete.diagnostics.lastBrowser, rustSettings);
  assert.deepEqual(audit.before.view, {
    rustHidden: true,
    browserHidden: false,
    rustInputCount: 0,
  });
} else {
  assert.equal(audit.before.diagnostics.state, 'disabled');
  assert.equal(audit.complete.diagnostics.dispatches, 0);
  assert.equal(audit.complete.diagnostics.comparisons, 0);
  assert.equal(audit.complete.diagnostics.authorityWrites, 0);
  assert.equal(audit.complete.diagnostics.viewMountAttempts, 0);
  assert.deepEqual(audit.before.view, {
    rustHidden: true,
    browserHidden: false,
    rustInputCount: 0,
  });
}
