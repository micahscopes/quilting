import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const { default: init, canonicalizeHyperscopeRoute } = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

assert.deepEqual(canonicalizeHyperscopeRoute([]).primaryAssetSettings, {
  uri: 'horse.glb',
  playing: true,
});
assert.deepEqual(canonicalizeHyperscopeRoute([]).startupSettings.primaryAssetSettings, {
  uri: 'horse.glb',
  playing: true,
});

const explicit = canonicalizeHyperscopeRoute([
  ['glb', 'local-glbs/classic_chessboard.glb'],
  ['anim', '17'],
  ['animate', '0'],
]);
assert.deepEqual(explicit.diagnostics, []);
assert.deepEqual(explicit.primaryAssetSettings, {
  uri: 'local-glbs/classic_chessboard.glb',
  animationClip: 17,
  playing: false,
});
assert.deepEqual(explicit.startupSettings.primaryAssetSettings, explicit.primaryAssetSettings);

const browser = readFileSync(`${repository}/hyperscope.html`, 'utf8');
for (const required of [
  '&& startupRoute.startupSettings',
  'initRustStartupSettings = startupRoute.startupSettings;',
  'currentGlb = initRustStartupSettings.primaryAssetSettings.uri;',
  'validatedPrimaryAssetSettings?.playing ?? params.animate',
  '? (initRustStartupSettings.primaryAssetSettings.animationClip ?? -1)',
]) {
  assert.ok(browser.includes(required), `browser typed primary asset route is missing ${required}`);
}

console.log('Rust typed primary asset route smoke passed');
