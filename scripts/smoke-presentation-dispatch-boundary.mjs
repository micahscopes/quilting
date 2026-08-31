import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const app = read('crates/hyperscope-app/src/lib.rs');
const web = read('crates/hyperscope-web/src/presentation_card.rs');
const csr = read('crates/hyperscope-web/src/presentation_card/csr.rs');
const adapter = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');

for (const required of [
  'pub struct AnimationClipEffects',
  'pub fn from_commit(commit: &AppCommit) -> Self',
  'pub struct PresentationDispatch',
  'pub fn dispatch_presentation(',
  'SemanticAction::Present(action)',
  'let effects = AnimationClipEffects::from_commit(&commit);',
  'selection: effects.selection',
  'cancellations: effects.cancellations',
]) {
  assert.ok(app.includes(required), `application presentation port is missing ${required}`);
}

for (const required of [
  'PresentationDispatch as PresentationCardCommit',
  'store.dispatch_presentation(action.semantic())',
]) {
  assert.ok(web.includes(required), `Leptos presentation adapter is missing ${required}`);
}
const webDispatch = web.slice(
  web.indexOf('pub fn activate_presentation_card('),
  web.indexOf('\n}\n\nimpl PresentationCardViewModel', web.indexOf('pub fn activate_presentation_card(')) + 2,
);
for (const retired of ['commit.effects', 'AppEffect::', 'for effect in']) {
  assert.equal(webDispatch.includes(retired), false,
    `hyperscope-web must not interpret presentation effects through ${retired}`);
}
for (const required of [
  'committed.commit.revision',
  'effect.scene_request_id.to_string()',
  'effect.asset_id.to_string()',
]) {
  assert.ok(csr.includes(required), `CSR presentation projection is missing ${required}`);
}

for (const required of [
  '#[wasm_bindgen(js_name = dispatchPresentation)]',
  '#[wasm_bindgen(js_name = requestPresentation)]',
  '.dispatch_presentation(presentation_action_from_wire(action, cue_id)?)',
  'ShadowPresentationDispatch',
  'ShadowAnimationClipJobEffect::selection',
  'ShadowAnimationClipJobEffect::cancellation',
]) {
  assert.ok(adapter.includes(required), `WASM presentation port is missing ${required}`);
}

const dispatchStart = browser.indexOf('function dispatchAppPresentation(direction, cueId) {');
const dispatchEnd = browser.indexOf('function mirrorAppPresentation(', dispatchStart);
assert.ok(dispatchStart >= 0 && dispatchEnd > dispatchStart,
  'could not locate browser presentation dispatch adapter');
const dispatch = browser.slice(dispatchStart, dispatchEnd);
for (const required of [
  'rustAppShadow.requestPresentation(direction, cueId || \'\')',
  'clipJob = { effect: receipt.selection, cancellations: receipt.cancellations };',
  'rustAppShadow.present(',
]) {
  assert.ok(dispatch.includes(required), `thin presentation adapter is missing ${required}`);
}
for (const retired of [
  'rustAppShadow.dispatchPresentation(direction, cueId',
  'receipt.commit.effects',
  'effect.type ===',
]) {
  assert.equal(dispatch.includes(retired), false,
    `ordinary presentation dispatch must not retain ${retired}`);
}

const effectsStart = browser.indexOf('function observePresentationClipJob(job, context) {');
const effectsEnd = browser.indexOf(
  'async function bindPrimaryPresentationAnimationResidency(',
  effectsStart,
);
assert.ok(effectsStart >= 0 && effectsEnd > effectsStart,
  'could not locate typed presentation clip observer');
const effects = browser.slice(effectsStart, effectsEnd);
for (const required of [
  'const effect = job?.effect || null;',
  'const cancellations = job?.cancellations || [];',
  'committedClipJob: job,',
]) {
  assert.ok(effects.includes(required), `typed presentation clip observer is missing ${required}`);
}
for (const retired of [
  'commit.effects',
  '.filter(',
  "effect.type === 'select_animation_clip'",
  "effect.type === 'cancel_animation_clip_selection'",
]) {
  assert.equal(effects.includes(retired), false,
    `browser presentation semantics must not retain ${retired}`);
}
assert.equal(browser.includes('function committedPresentationClipJob('), false,
  'browser presentation semantics must not retain committedPresentationClipJob');

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Presentation-dispatch Rust boundary source smoke passed');
