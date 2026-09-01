#!/usr/bin/env node

import assert from 'node:assert/strict';

const cdpEndpoint = process.env.HYPERSCOPE_CDP_ENDPOINT || 'http://127.0.0.1:9222';
const pagePort = process.env.HYPERSCOPE_PORT || '8888';
const targets = await (await fetch(`${cdpEndpoint}/json/list`)).json();
const originalPage = targets.find(target =>
  target.type === 'page' && target.url.includes(`:${pagePort}/`));
assert.ok(originalPage, 'a pre-existing Hyperscope page is required');

const route = new URL(process.env.HYPERSCOPE_WEBGL_ERROR_URL
  || `http://127.0.0.1:${pagePort}/`);
route.searchParams.set('_gl_error_audit', String(Date.now()));
const traceCalls = process.env.HYPERSCOPE_TRACE_WEBGL_ERRORS === '1';

const page = await (await fetch(
  `${cdpEndpoint}/json/new?${encodeURIComponent('about:blank')}`,
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
socket.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  const waiter = pending.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
  else waiter.resolve(message.result);
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
  });
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.exception?.description
      || response.exceptionDetails.text);
  }
  return response.result.value;
}

try {
  await command('Runtime.enable');
  await command('Page.enable');
  if (traceCalls) {
    await command('Page.addScriptToEvaluateOnNewDocument', {
      source: `(() => {
        const errors = [];
        const recentCalls = [];
        globalThis.__hyperscopeWebGlErrors = errors;
        const getError = WebGL2RenderingContext.prototype.getError;
        const seen = new Set();
        const describe = value => {
          if (value == null || ['boolean', 'number', 'string'].includes(typeof value)) {
            return value;
          }
          if (ArrayBuffer.isView(value)) {
            return value.constructor.name + '[' + value.length + ']';
          }
          return value.constructor?.name || typeof value;
        };
        for (let prototype = WebGL2RenderingContext.prototype;
          prototype && prototype !== Object.prototype;
          prototype = Object.getPrototypeOf(prototype)) {
          for (const name of Object.getOwnPropertyNames(prototype)) {
            if (name === 'constructor' || name === 'getError' || seen.has(name)) continue;
            const descriptor = Object.getOwnPropertyDescriptor(prototype, name);
            if (!descriptor || typeof descriptor.value !== 'function') continue;
            seen.add(name);
            const original = descriptor.value;
            try {
              Object.defineProperty(prototype, name, {
                ...descriptor,
                value: function (...args) {
                  const call = { method: name, args: args.map(describe) };
                  recentCalls.push(call);
                  if (recentCalls.length > 24) recentCalls.shift();
                  const result = original.apply(this, args);
                  const error = getError.call(this);
                  if (error !== 0 && errors.length < 64) {
                    errors.push({
                      method: name,
                      error,
                      args: args.map(describe),
                      recentCalls: recentCalls.map(entry => ({ ...entry })),
                      stack: new Error('WebGL error').stack,
                    });
                  }
                  return result;
                },
              });
            } catch (_) {}
          }
        }
      })();`,
    });
  }
  await command('Page.navigate', { url: route.href });
  const snapshot = await evaluate(`(async () => {
    let state = null;
    for (let attempt = 0; attempt < 300; attempt += 1) {
      const graphics = globalThis.__hyperscopeGraphicsBackend;
      const residency = await graphics?.refresh?.();
      state = {
        url: location.href,
        phase: globalThis.__hyperscopeStartup?.state ?? null,
        graphics: graphics ? {
          effective: graphics.effective,
          state: graphics.state,
          error: graphics.error ?? null,
        } : null,
        residency: residency ? {
          state: residency.state,
          modelFaces: Number(residency.modelFaces || 0),
          presentationFrames: Number(residency.presentationFrames || 0),
          frameFailures: Number(residency.frameFailures || 0),
        } : null,
      };
      if (state.residency?.modelFaces > 0
          && (state.graphics.effective === 'webgpu'
            || document.getElementById('stats')?.textContent?.includes('faces'))) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    await new Promise(resolve => setTimeout(resolve, 500));
    const gl = document.getElementById('cv')?.getContext('webgl2');
    return {
      ...state,
      error: gl?.getError() ?? -1,
      tracedErrors: [...(globalThis.__hyperscopeWebGlErrors || [])],
      stats: document.getElementById('stats')?.textContent ?? null,
    };
  })()`);
  console.log(JSON.stringify(snapshot));
  assert.deepEqual(snapshot.tracedErrors, [], 'WebGL call tracing found an error');
  assert.equal(
    snapshot.error,
    0,
    `WebGL context retained error 0x${snapshot.error.toString(16)} on ${snapshot.url}`,
  );
} finally {
  await activate(originalPage.id);
  socket.close();
  await fetch(`${cdpEndpoint}/json/close/${page.id}`);
}
