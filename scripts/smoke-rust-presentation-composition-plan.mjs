import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const app = read('crates/hyperscope-app/src/lib.rs');
const bridge = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');

assert.ok(app.includes('pub struct PresentationCompositionPlan {'),
  'hyperscope-app must own stable composition identity');
for (const policy of [
  'presentation_animation_residency',
  'asset.id == resident_id.as_uuid()',
  'presentation_uri_leaf(&asset.uri) == resident_leaf',
  'AmbiguousPrimaryAsset',
  'PresentationAssetResolutionError',
  'presentation_asset_for_exact_uri',
  'AmbiguousExactUri',
]) {
  assert.ok(app.includes(policy), `composition plan is missing policy ${policy}`);
}
assert.ok(bridge.includes('#[wasm_bindgen(js_name = presentationCompositionPlan)]'),
  'the WASM adapter must expose the typed composition plan');
assert.ok(bridge.includes('#[wasm_bindgen(js_name = presentationAssetForExactUri)]'),
  'the WASM adapter must expose exact presentation asset identity');

const fetchResolverStart = browser.indexOf('function durablePresentationFetchAssetId(');
const fetchResolverEnd = browser.indexOf('function recordAppShadowMismatch(', fetchResolverStart);
const fetchResolver = browser.slice(fetchResolverStart, fetchResolverEnd);
assert.ok(fetchResolverStart >= 0 && fetchResolverEnd > fetchResolverStart,
  'could not locate presentation fetch identity adapter');
for (const required of [
  "if (provenance !== 'manifest-fetch') return null;",
  'if (presentationAppAuthority()) {',
  'rustAppShadow.presentationAssetForExactUri(String(requestedUri))',
  'durablePresentationAssetId(',
]) {
  assert.ok(fetchResolver.includes(required),
    `presentation fetch identity adapter is missing ${required}`);
}

const modelPhaseStart = browser.indexOf("phase('model', ['workers'], async () => {");
const modelPhaseEnd = browser.indexOf("phase('render',", modelPhaseStart);
const modelPhase = browser.slice(modelPhaseStart, modelPhaseEnd);
assert.ok(modelPhase.includes('durablePresentationFetchAssetId('),
  'startup fetch completion must use the authority-aware identity resolver');
assert.equal(modelPhase.includes('durablePresentationAssetId('), false,
  'Rust startup must not resolve durable presentation identity in JavaScript');

const start = browser.indexOf('async function initializePresentationComposition()');
const end = browser.indexOf('\n// ============================================================', start);
const composition = browser.slice(start, end);
assert.ok(composition.includes('rustAppShadow.presentationCompositionPlan()'),
  'Rust presentation authority must provide composition identity');
assert.ok(composition.includes('const primaryAsset = rustPlan?.primary'),
  'the primary renderer residency must adapt the Rust plan');
assert.ok(composition.includes('const secondaryAssets = rustPlan?.secondary'),
  'secondary renderer residency must preserve Rust plan order');
assert.ok(composition.includes('presentationAssetMatchesFilename(asset, currentGlb)'),
  'the standalone js|shadow rollback resolver must remain available');

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Rust presentation-composition plan source smoke passed');
