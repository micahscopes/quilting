import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const app = read('crates/hyperscope-app/src/lib.rs');
const adapter = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');

for (const required of [
  'pub struct PrimarySceneInstallCompletionDispatch',
  'pub fn complete_primary_scene_install(',
  'EffectCompletion::PrimarySceneInstall(completion)',
  'let jobs = AnimationClipEffects::from_commit(&commit);',
  'clip_cancellations: jobs.cancellations',
  'fn typed_scene_install_completion_exposes_obsolete_clip_cancellation()',
]) {
  assert.ok(app.includes(required), `typed primary-install port is missing ${required}`);
}

for (const required of [
  '#[wasm_bindgen(js_name = completePrimarySceneInstalled)]',
  '#[wasm_bindgen(js_name = completePrimarySceneInstallFailed)]',
  '#[wasm_bindgen(js_name = finishPrimarySceneInstalled)]',
  '#[wasm_bindgen(js_name = finishPrimarySceneInstallFailed)]',
  '.complete_primary_scene_install(PrimarySceneInstallCompletion {',
  'struct ShadowPrimarySceneInstallCompletionDispatch',
  'clip_cancellations: dispatch',
  '.map(ShadowAnimationClipJobEffect::cancellation)',
]) {
  assert.ok(adapter.includes(required), `WASM primary-install port is missing ${required}`);
}

const successStart = browser.indexOf('function completeAppPrimarySceneInstall(');
const successEnd = browser.indexOf('function failAppPrimarySceneInstall(', successStart);
const success = browser.slice(successStart, successEnd);
for (const required of [
  'rustAppShadow.finishPrimarySceneInstalled(',
  'const commit = receipt.commit;',
  'receipt.clipCancellations',
  'observePrimarySceneInstallClipCancellations(',
]) {
  assert.ok(success.includes(required), `primary-install success adapter is missing ${required}`);
}
assert.equal(success.includes('rustAppShadow.completePrimarySceneInstalled('), false,
  'ordinary primary-install success must not use the generic completion seam');

const failureStart = browser.indexOf('function failAppPrimarySceneInstall(');
const failureEnd = browser.indexOf('function failAppAssetShadow(', failureStart);
const failure = browser.slice(failureStart, failureEnd);
for (const required of [
  'rustAppShadow.finishPrimarySceneInstallFailed(',
  'const commit = receipt.commit;',
  'receipt.clipCancellations',
]) {
  assert.ok(failure.includes(required), `primary-install failure adapter is missing ${required}`);
}
assert.equal(failure.includes('rustAppShadow.completePrimarySceneInstallFailed('), false,
  'ordinary primary-install failure must not use the generic completion seam');

const observerStart = browser.indexOf('function observePrimarySceneInstallClipCancellations(');
const observerEnd = browser.indexOf('function completeAppPrimarySceneInstall(', observerStart);
const observer = browser.slice(observerStart, observerEnd);
for (const required of [
  "cancellation.type !== 'cancel_animation_clip_selection'",
  'animationPoseBlockGeneration += 1;',
  'presentationAnimationGeneration += 1;',
  'animationPoseBlocked = false;',
]) {
  assert.ok(observer.includes(required), `scene-replacement cancellation fence is missing ${required}`);
}

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Primary-scene install typed-completion source smoke passed');
