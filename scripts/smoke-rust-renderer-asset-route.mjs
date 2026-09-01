import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const { default: init, canonicalizeHyperscopeRoute } = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

assert.deepEqual(canonicalizeHyperscopeRoute([]).rendererAssetSettings, {
  environment: 'rosendal_plains_1_1k',
  matcap: 'citric-acid',
});

const explicit = canonicalizeHyperscopeRoute([
  ['env', 'rogland_clear_night_2k'],
  ['matcap', 'soft-studio'],
]);
assert.deepEqual(explicit.diagnostics, []);
assert.deepEqual(explicit.rendererAssetSettings, {
  environment: 'rogland_clear_night_2k',
  matcap: 'soft-studio',
});

for (const key of ['env', 'matcap']) {
  const invalid = canonicalizeHyperscopeRoute([[key, '']]);
  assert.ok(invalid.diagnostics.some(diagnostic =>
    diagnostic.key === key && diagnostic.code === 'invalid_value'));
  assert.equal(invalid.rendererAssetSettings, undefined);
}

const browser = readFileSync(`${repository}/hyperscope.html`, 'utf8');
for (const required of [
  '&& startupRoute.rendererAssetSettings',
  'initRendererAssetSettings = startupRoute.rendererAssetSettings;',
  'validatedRendererAssetSettings?.environment',
  'validatedRendererAssetSettings?.matcap',
]) {
  assert.ok(browser.includes(required), `browser typed renderer asset route is missing ${required}`);
}

console.log('Rust typed renderer asset route smoke passed');
