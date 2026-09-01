import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const app = read('crates/hyperscope-app/src/lib.rs');
const adapter = read('crates/quilting-wasm/src/app_shadow.rs');
const loader = read('crates/quilting-wasm/src/lib.rs');
const browser = read('hyperscope.html');

for (const required of [
  'pub fn reconcile_primary_scene_asset_identity(',
  'PrimarySceneAssetIdentityProvenance::EmbeddedAuthoring',
  'fn primary_scene_identity_rejects_conflicting_durable_declarations()',
  'pub struct PrimarySceneInstallCompletionDispatch',
  'pub fn complete_primary_scene_install(',
  'EffectCompletion::PrimarySceneInstall(completion)',
  'let jobs = AnimationClipEffects::from_commit(&commit);',
  'clip_cancellations: jobs.cancellations',
  'installed_scene: self.installed_primary_scene_snapshot()',
  'clip_state: self.animation_clip_selection_snapshot()',
  'fn typed_scene_install_completion_exposes_obsolete_clip_cancellation()',
]) {
  assert.ok(app.includes(required), `typed primary-install port is missing ${required}`);
}

for (const required of [
  '#[wasm_bindgen(js_name = reconcilePrimarySceneAssetIdentity)]',
  'struct ShadowPrimarySceneAssetIdentity',
  'primary_scene_asset_identity_to_js(resolution)',
  '#[wasm_bindgen(js_name = completePrimarySceneInstalled)]',
  '#[wasm_bindgen(js_name = completePrimarySceneInstallFailed)]',
  '#[wasm_bindgen(js_name = finishPrimarySceneInstalled)]',
  '#[wasm_bindgen(js_name = finishPrimarySceneInstallFailed)]',
  '.complete_primary_scene_install(PrimarySceneInstallCompletion {',
  'struct ShadowPrimarySceneInstallCompletionDispatch',
  'clip_cancellations: dispatch',
  '.map(ShadowAnimationClipJobEffect::cancellation)',
  'installed_scene: dispatch.installed_scene.map(Into::into)',
  'clip_state: dispatch.clip_state.into()',
]) {
  assert.ok(adapter.includes(required), `WASM primary-install port is missing ${required}`);
}
for (const required of [
  '&"hyperscape_asset_id".into()',
  '.and_then(|asset| asset.payload.asset_id)',
]) {
  assert.ok(loader.includes(required), `glTF identity extraction is missing ${required}`);
}

const reconcileStart = browser.indexOf('function reconcileDecodedPrimarySceneAssetIdentity(');
const reconcileEnd = browser.indexOf(
  'function observePrimarySceneInstallClipCancellations(',
  reconcileStart,
);
const reconcile = browser.slice(reconcileStart, reconcileEnd);
for (const required of [
  'rustAppShadow.reconcilePrimarySceneAssetIdentity(',
  'info?.hyperscape_asset_id',
  'resolution.interactionAssetId',
  'resolution.authoringAssetId',
]) {
  assert.ok(reconcile.includes(required), `primary-scene identity join is missing ${required}`);
}

const uploadStart = browser.indexOf('async function finishModelUpload(');
const uploadEnd = browser.indexOf('async function loadStaticPresentationAsset(', uploadStart);
const upload = browser.slice(uploadStart, uploadEnd);
const reconciliationOffset = upload.indexOf('reconcileDecodedPrimarySceneAssetIdentity(');
const rendererMutationOffset = upload.indexOf('animationPoseBlockGeneration += 1;');
assert.ok(reconciliationOffset >= 0, 'primary renderer upload must reconcile asset identity');
assert.ok(rendererMutationOffset > reconciliationOffset,
  'asset identity must reconcile before primary renderer state is mutated');
assert.ok(upload.includes('if (currentModelDurableAssetId && modelNodeStableEntityIds.some(Boolean))'),
  'stable node UUIDs must require reconciled durable asset identity');
assert.equal(
  upload.includes('const currentIdentityAssetId = currentModelDurableAssetId || currentModelSessionAssetId;'),
  false,
  'session residency must not promote stable node UUIDs to durable identity',
);

const stableInstallStart = browser.indexOf('function installCurrentModelStableIdentities(');
const stableInstallEnd = browser.indexOf('function applyLocalPeerAuthoredProjection(', stableInstallStart);
const stableInstall = browser.slice(stableInstallStart, stableInstallEnd);
assert.ok(stableInstall.includes('!currentModelDurableAssetId'),
  'authored projection must not reinstall stable UUIDs without durable asset scope');

const secondaryStart = browser.indexOf('async function loadStaticPresentationAsset(');
const secondaryEnd = browser.indexOf('function extendPrimaryAnimationTextures(', secondaryStart);
const secondary = browser.slice(secondaryStart, secondaryEnd);
assert.ok(secondary.includes('reconcileDecodedPrimarySceneAssetIdentity('),
  'secondary presentation assets must use the same durable identity gate');
assert.ok(secondary.includes('assetIdentity.authoringAssetId !== String(asset.id).toLowerCase()'),
  'secondary presentation identity must remain equal to its catalog declaration');

const successStart = browser.indexOf('function completeAppPrimarySceneInstall(');
const successEnd = browser.indexOf('function failAppPrimarySceneInstall(', successStart);
const success = browser.slice(successStart, successEnd);
for (const required of [
  'rustAppShadow.finishPrimarySceneInstalled(',
  'const commit = receipt.commit;',
  'receipt.clipCancellations',
  'installedPrimaryScene: receipt.installedScene',
  'animationClipSelection: receipt.clipState',
  'observePrimarySceneInstallClipCancellations(',
]) {
  assert.ok(success.includes(required), `primary-install success adapter is missing ${required}`);
}
assert.equal(success.includes('rustAppShadow.completePrimarySceneInstalled('), false,
  'ordinary primary-install success must not use the generic completion seam');
assert.equal(success.includes('refreshAppShadowSnapshot()'), false,
  'primary-install success must consume its compact typed state');

const failureStart = browser.indexOf('function failAppPrimarySceneInstall(');
const failureEnd = browser.indexOf('function failAppAssetShadow(', failureStart);
const failure = browser.slice(failureStart, failureEnd);
for (const required of [
  'rustAppShadow.finishPrimarySceneInstallFailed(',
  'const commit = receipt.commit;',
  'receipt.clipCancellations',
  'installedPrimaryScene: receipt.installedScene',
  'animationClipSelection: receipt.clipState',
]) {
  assert.ok(failure.includes(required), `primary-install failure adapter is missing ${required}`);
}
assert.equal(failure.includes('rustAppShadow.completePrimarySceneInstallFailed('), false,
  'ordinary primary-install failure must not use the generic completion seam');
assert.equal(failure.includes('refreshAppShadowSnapshot()'), false,
  'primary-install failure must consume its compact typed state');

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
