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

const browser = readFileSync(`${repository}/hyperscope.html`, 'utf8');
for (const required of [
  '&& startupRoute.primaryAssetSettings',
  'initPrimaryAssetSettings = startupRoute.primaryAssetSettings;',
  'currentGlb = initPrimaryAssetSettings.uri;',
  'validatedPrimaryAssetSettings?.playing ?? params.animate',
  '? (initPrimaryAssetSettings?.animationClip ?? -1)',
]) {
  assert.ok(browser.includes(required), `browser typed primary asset route is missing ${required}`);
}

console.log('Rust typed primary asset route smoke passed');
