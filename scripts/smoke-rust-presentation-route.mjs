import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const { default: init, canonicalizeHyperscopeRoute } = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

assert.deepEqual(canonicalizeHyperscopeRoute([]).presentationSettings, {
  enabled: false,
});

const cueId = 'e0000000-0000-4000-8000-000000000004';
const linked = canonicalizeHyperscopeRoute([
  ['presentation', '1'],
  ['cue', cueId],
]);
assert.deepEqual(linked.diagnostics, []);
assert.deepEqual(linked.presentationSettings, {
  enabled: true,
  cueId,
});

const misleading = canonicalizeHyperscopeRoute([['cue', cueId]]);
assert.ok(misleading.diagnostics.some(diagnostic =>
  diagnostic.key === 'presentation' && diagnostic.code === 'invalid_value'));
assert.equal(misleading.presentationSettings, undefined);

const browser = readFileSync(`${repository}/hyperscope.html`, 'utf8');
for (const required of [
  '&& startupRoute.presentationSettings',
  'initPresentationSettings = startupRoute.presentationSettings;',
  'initPresentationSettings.enabled !== RUST_PRESENTATION_ENABLED',
  '? initPresentationSettings?.cueId : initParams.cue;',
  "? activateRustPresentation('jump', routeCueId)",
]) {
  assert.ok(browser.includes(required), `browser typed presentation route is missing ${required}`);
}

console.log('Rust typed presentation route smoke passed');
