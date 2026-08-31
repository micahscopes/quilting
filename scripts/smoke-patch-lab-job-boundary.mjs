import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const app = read('crates/hyperscope-app/src/lib.rs');
const patchLab = read('crates/hyperscope-app/src/patch_lab.rs');
const patchLabWeb = read('crates/hyperscope-web/src/patch_lab.rs');
const renderWeb = read('crates/hyperscope-web/src/render_controls.rs');
const adapter = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');

for (const required of [
  'pub struct PatchLabSessionDispatch',
  'pub struct PatchLabCompletionDispatch',
  'pub fn from_effects(effects: &[AppEffect]) -> Self',
  'pub fn set_patch_lab_session(',
  'pub fn complete_patch_lab(',
  'let effects = PatchLabEffects::from_commit(&commit);',
]) {
  assert.ok(app.includes(required) || patchLab.includes(required),
    `application Patch Lab job port is missing ${required}`);
}

for (const [label, source] of [
  ['Patch Lab controls', patchLabWeb],
  ['render controls', renderWeb],
]) {
  assert.equal(source.includes('AppEffect::PatchLab'), false,
    `${label} must not parse top-level application effects`);
}
assert.ok(patchLabWeb.includes('store.set_patch_lab_session(intent)?'),
  'Patch Lab controls must delegate to the typed AppStore port');
assert.ok(renderWeb.includes('PatchLabEffects::from_commit(&commit).into_vec()'),
  'render controls must use the shared Patch Lab projection');

for (const required of [
  '#[wasm_bindgen(js_name = dispatchPatchLab)]',
  '#[wasm_bindgen(js_name = requestPatchLab)]',
  '#[wasm_bindgen(js_name = finishPatchLabGeometry)]',
  '#[wasm_bindgen(js_name = finishPatchLabGeometryFailed)]',
  '#[wasm_bindgen(js_name = finishPatchLabLod)]',
  '#[wasm_bindgen(js_name = finishPatchLabLodFailed)]',
  '#[wasm_bindgen(js_name = drainPatchLabEffects)]',
  'patch_lab_effects: shadow_patch_lab_effects(&patch_lab_effects)',
  'self.store.complete_patch_lab(completion).map_err(js_error)',
]) {
  assert.ok(adapter.includes(required), `WASM Patch Lab job port is missing ${required}`);
}

const synchronizationStart = browser.indexOf('function synchronizeRustPatchLabFromBrowser(');
const synchronizationEnd = browser.indexOf('function patchLabHistogramBins(', synchronizationStart);
const synchronization = browser.slice(synchronizationStart, synchronizationEnd);
for (const required of [
  'app.requestPatchLab(browser)',
  'observeRustPatchLabEffects(receipt.effects, context)',
]) {
  assert.ok(synchronization.includes(required), `Patch Lab request adapter is missing ${required}`);
}

const completionStart = browser.indexOf('function completeRustPatchLabGeometryFromBrowser(');
const completionEnd = browser.indexOf('function backendNeutralRenderStyle(', completionStart);
const completion = browser.slice(completionStart, completionEnd);
for (const required of [
  'rustAppShadow.finishPatchLabGeometry(',
  'rustAppShadow.finishPatchLabLod(',
  'rustAppShadow.finishPatchLabGeometryFailed(',
  'rustAppShadow.finishPatchLabLodFailed(',
  'observeRustPatchLabEffects(receipt.effects, context)',
]) {
  assert.ok(completion.includes(required), `Patch Lab completion adapter is missing ${required}`);
}

const observerStart = browser.indexOf('function observeRustPatchLabEffects(');
const observerEnd = browser.indexOf('function rustPatchLabControlProjection(', observerStart);
const observer = browser.slice(observerStart, observerEnd);
for (const required of [
  'for (const effect of effects)',
  "case 'build_geometry':",
  "case 'evaluate_lod':",
  'returned an invalid Patch Lab job',
]) {
  assert.ok(observer.includes(required), `typed Patch Lab executor is missing ${required}`);
}
for (const retired of [
  'unwrapRustPatchLabEffect',
  'rawEffect',
  'patchTypes',
  '.filter(',
]) {
  assert.equal(observer.includes(retired), false,
    `typed Patch Lab executor must not retain ${retired}`);
}

for (const retired of [
  'commit.effects',
  'receipt.commit.effects',
  'drainAdapterEffects()',
  'app.dispatchPatchLab(browser)',
]) {
  assert.equal(browser.includes(retired), false,
    `ordinary browser Patch Lab paths must not retain ${retired}`);
}
for (const required of [
  'receipt.patchLabEffects',
  'rustAppShadow.drainPatchLabEffects()',
]) {
  assert.ok(browser.includes(required), `typed Patch Lab browser path is missing ${required}`);
}

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Patch Lab typed-job boundary source smoke passed');
