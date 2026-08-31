import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const app = read('crates/hyperscope-app/src/lib.rs');
const web = read('crates/hyperscope-web/src/animation_control.rs');
const csr = read('crates/hyperscope-web/src/animation_control/csr.rs');
const adapter = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');
const settings = read('crates/hyperscope-app/src/settings.rs');

for (const required of [
  'spec!("animclipimpl", "js", Implementation)',
  'pub struct AnimationClipJobEffect',
  'pub struct AnimationClipRequest',
  'pub struct AnimationClipCompletionDispatch',
  'pub fn request_animation_clip(',
  'pub fn complete_animation_clip_selection(',
  'SemanticAction::Animate(AnimationAction::SelectClip(index))',
  'matches_request: selected_index == Some(index)',
  'state: AnimationClipSelectionReadModel',
]) {
  assert.ok(settings.includes(required) || app.includes(required),
    `application clip-request port is missing ${required}`);
}

for (const required of [
  'AnimationClipRequest as AnimationClipControlCommit',
  'store.request_animation_clip(index)',
]) {
  assert.ok(web.includes(required), `Leptos clip adapter is missing ${required}`);
}
const webSelection = web.slice(
  web.indexOf('pub fn select_animation_clip('),
  web.indexOf('\n}\n\n#[cfg(test)]', web.indexOf('pub fn select_animation_clip(')) + 2,
);
for (const retired of ['commit.effects', 'AppEffect::SelectAnimationClip', 'for effect in']) {
  assert.equal(webSelection.includes(retired), false,
    `hyperscope-web must not parse reducer effects through ${retired}`);
}
for (const required of [
  'committed.commit.revision.to_string()',
  '.map(|active| f64::from(active.clip.index))',
  '.map(|pending| f64::from(pending.clip.index))',
  'effect.scene_request_id.to_string()',
  'effect.asset_id.to_string()',
]) {
  assert.ok(csr.includes(required), `CSR clip projection is missing ${required}`);
}

for (const required of [
  '#[wasm_bindgen(js_name = dispatchAnimationClip)]',
  '#[wasm_bindgen(js_name = requestAnimationClip)]',
  '#[wasm_bindgen(js_name = completeAnimationClipSelected)]',
  '#[wasm_bindgen(js_name = completeAnimationClipSelectionFailed)]',
  '#[wasm_bindgen(js_name = finishAnimationClipSelected)]',
  '#[wasm_bindgen(js_name = finishAnimationClipSelectionFailed)]',
  'let request = self.store.request_animation_clip(index)',
  'ShadowAnimationClipJobEffect::selection',
  'ShadowAnimationClipJobEffect::cancellation',
  'matches_request: request.matches_request',
  'state: request.state.into()',
  'struct ShadowAnimationClipCompletionDispatch',
  'selection: dispatch.state.into()',
]) {
  assert.ok(adapter.includes(required), `WASM clip-request port is missing ${required}`);
}

const requestStart = browser.indexOf('function beginAppAnimationClipSelection(index) {');
const requestEnd = browser.indexOf('function completeAppAnimationClipSelection(', requestStart);
assert.ok(requestStart >= 0 && requestEnd > requestStart,
  'could not locate browser clip-request adapter');
const request = browser.slice(requestStart, requestEnd);
for (const required of [
  'app.requestAnimationClip(index)',
  'const effect = receipt.selection;',
  'const cancellations = receipt.cancellations;',
  'const residencySnapshot = { animationClipSelection: receipt.state };',
  'if (!receipt.matchesRequest)',
]) {
  assert.ok(request.includes(required), `thin browser clip adapter is missing ${required}`);
}
for (const retired of [
  'app.dispatchAnimationClip(index)',
  'receipt.commit.effects.filter(',
  "effect.type === 'select_animation_clip'",
  "effect.type === 'cancel_animation_clip_selection'",
  'snapshot?.animationClipSelection?.active?.clip?.index',
  'refreshAppShadowSnapshot()',
]) {
  assert.equal(request.includes(retired), false,
    `browser clip adapter must not retain ${retired}`);
}

const controlStart = browser.indexOf('function ensureRustAnimationClipControlView() {');
const controlEnd = browser.indexOf('function ensureRustNavigationStatusView()', controlStart);
const control = browser.slice(controlStart, controlEnd);
for (const required of [
  '(index, sequence, revision, activeIndex, pendingIndex, effect, cancellations)',
  'active: Number.isSafeInteger(active) && active >= 0',
  'pending: Number.isSafeInteger(pending) && pending >= 0',
  'observeAnimationClipResidency(',
]) {
  assert.ok(control.includes(required), `Leptos clip callback is missing ${required}`);
}
assert.equal(control.includes('refreshAppShadowSnapshot()'), false,
  'Leptos clip callbacks must not serialize the complete application state');

const completionStart = browser.indexOf('function completeAppAnimationClipSelection(');
const completionEnd = browser.indexOf('async function selectAnimationIndex(', completionStart);
const completion = browser.slice(completionStart, completionEnd);
for (const required of [
  'rustAppShadow.finishAnimationClipSelected(',
  'rustAppShadow.finishAnimationClipSelectionFailed(',
  'const commit = receipt.commit;',
  '{ animationClipSelection: receipt.selection }',
]) {
  assert.ok(completion.includes(required),
    `typed browser clip-completion adapter is missing ${required}`);
}
for (const retired of [
  'rustAppShadow.completeAnimationClipSelected(',
  'rustAppShadow.completeAnimationClipSelectionFailed(',
  'refreshAppShadowSnapshot()',
]) {
  assert.equal(completion.includes(retired), false,
    `ordinary clip completion must not retain ${retired}`);
}

const rangeStart = browser.indexOf('function animationClipRangeForClock(app) {');
const rangeEnd = browser.indexOf('function writeRustAnimationSample(', rangeStart);
const range = browser.slice(rangeStart, rangeEnd);
assert.ok(range.includes('app?.animationRuntimeState()?.clipState?.active?.clip'),
  'clock range lookup must use the compact Rust animation runtime state');
assert.equal(range.includes('.snapshot()'), false,
  'clock range lookup must not serialize the complete application state');

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Animation-clip Rust boundary source smoke passed');
