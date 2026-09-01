import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const browser = read('hyperscope.html');
const appShadow = read('crates/quilting-wasm/src/app_shadow.rs');
const renderer = read('crates/quilting-wasm/src/main_renderer.rs');
const webgpuBackend = read('crates/quilting-wasm/src/webgpu_backend.rs');
const renderEvidence = read('crates/quilting-core/src/render_evidence.rs');
const interaction = read('crates/hyperscape/src/interaction.rs');
const settings = read('crates/hyperscope-app/src/settings.rs');

for (const required of [
  'spec!("pickimpl", "js", Implementation)',
  'const PICK_IMPLEMENTATION = implementationFromRoute(',
  "initialBrowserParams, 'pickimpl'",
  "set('pickimpl', PICK_IMPLEMENTATION, PARAM_DEFAULTS.pickimpl)",
]) {
  assert.ok(
    settings.includes(required) || browser.includes(required),
    `pick shadow rollback route is missing ${required}`,
  );
}

for (const required of [
  'pub struct RenderPickEvidenceReport',
  'pub fn validate(self) -> Result<(), RenderPickEvidenceError>',
  'RenderPickComparison::between(',
  'if canonical != self.comparison',
]) {
  assert.ok(renderEvidence.includes(required), `shared pick evidence is missing ${required}`);
}

for (const required of [
  'pub struct InteractionPickEvidenceObserver',
  'report.target_epoch != targets.epoch()',
  'InteractionPickEvidenceDisposition::IgnoredStale',
  'self.diagnostics.last_report = Some(report)',
]) {
  assert.ok(interaction.includes(required), `Hyperscape pick observer is missing ${required}`);
}

for (const required of [
  'pub struct InteractionPickAuthority',
  'pub fn observe_readback(',
  'self.latest != Some(request)',
  'request.target_epoch != current_target_epoch',
  'InteractionPickAuthorityState::Resolving',
]) {
  assert.ok(interaction.includes(required), `Hyperscape pick authority is missing ${required}`);
}

for (const required of [
  '#[wasm_bindgen(js_name = stageBackendPickEvidence)]',
  '#[wasm_bindgen(js_name = readBackendPickEvidence)]',
  '.interaction_targets',
  'crate::main_renderer::stage_backend_pick_evidence(',
  'crate::main_renderer::read_backend_pick_evidence().await',
  '.record_report(&targets, report)',
]) {
  assert.ok(appShadow.includes(required), `AppShadow pick boundary is missing ${required}`);
}

for (const required of [
  '#[wasm_bindgen(js_name = pickBackendSurface)]',
  '#[wasm_bindgen(js_name = backendPickAuthorityDiagnostics)]',
  'crate::main_renderer::stage_backend_pick_authority(',
  '.observe_readback(request, current_target_epoch)',
  'crate::main_renderer::resolve_backend_pick_surface(raw)',
  '.publish(request, hit.expect("activation readiness requires a hit"))',
  '#[wasm_bindgen(js_name = activateBackendPick)]',
  'InteractionAction::ActivatePrimary(hit)',
  '#[wasm_bindgen(js_name = discardBackendPickActivation)]',
  '#[wasm_bindgen(js_name = clearSelection)]',
  'self.store.dispatch_selection_clear()',
  '#[wasm_bindgen(js_name = activatePackedInteraction)]',
  'self.activate_interaction_hit(hit)',
  '.accept(request)',
]) {
  assert.ok(appShadow.includes(required), `AppShadow pick authority is missing ${required}`);
}

for (const required of [
  'pub(crate) struct BackendPickEvidenceStageReceipt',
  'pub(crate) fn stage_backend_pick_evidence(',
  'pub(crate) async fn read_backend_pick_evidence()',
  'stage_backend_pick_evidence(mvp, mv, camera_pos, x, y, target_epoch).into_js()',
  'let report = read_backend_pick_evidence()',
  'let prior_highlight = STATE.with(',
  'state.highlight_face = prior_highlight;',
]) {
  assert.ok(renderer.includes(required), `renderer pick boundary is missing ${required}`);
}

for (const required of [
  'pub(crate) struct BackendPickAuthorityCapture',
  'pub(crate) fn stage_backend_pick_authority(',
  'pub(crate) fn resolve_backend_pick_surface(',
  'staged.source_render_call() != source_render_call',
  'staged.viewport() != viewport',
]) {
  assert.ok(renderer.includes(required), `renderer pick authority is missing ${required}`);
}

for (const required of [
  'last_completed_frame_input: Option<LiveFrameInput>',
  'pick_frame_ready: self.last_completed_frame_input.is_some()',
  'backend.last_completed_frame_input = Some(frame_input);',
  'backend.last_source_render_call = source_render_call;',
  '.last_completed_frame_input',
]) {
  assert.ok(
    webgpuBackend.includes(required),
    `WebGPU completed-frame pick residency is missing ${required}`,
  );
}

for (const required of [
  'pub(crate) struct StagedWebGpuPick',
  'viewport: backend.last_viewport,',
  'pub(crate) fn viewport(&self) -> [u32; 2]',
]) {
  assert.ok(webgpuBackend.includes(required), `WebGPU pick authority is missing ${required}`);
}

for (const required of [
  'const staged = app.stageBackendPickEvidence(',
  'Promise.resolve(app.readBackendPickEvidence())',
  'updateBackendPickDiagnostics(app.backendPickDiagnostics())',
  'return backendPickResult(staged?.webgl ?? null, true);',
]) {
  assert.ok(browser.includes(required), `browser pick adapter is missing ${required}`);
}


for (const required of [
  'async function pickSurfaceAtCanvasPixel(x, y, retainForActivation = false)',
  "if (PICK_IMPLEMENTATION === 'rust')",
  'result = await app.pickBackendSurface(x, y, retainForActivation);',
  "result?.disposition !== 'accepted'",
  "packedIdentity.assetId !== semanticIdentity.asset_id",
  "packedIdentity.entityId !== semanticIdentity.entity_id",
  "backendPickDiagnostics.adapterState = 'webgpu-authoritative';",
  'result.activationReady ? result.requestId : null,',
  'function activateSelectedObjectBackendPick(requestId, nowMs = performance.now())',
  'const navigation = rustAppShadow.activateBackendPick(requestId);',
  'function activateSelectedObjectPackedPick(nowMs = performance.now())',
  'const navigation = rustAppShadow.activatePackedInteraction(',
  ': activateSelectedObjectPackedPick(selectedAtMs);',
  'function commitSelectedObjectInteractionActivation(',
  '? activateSelectedObjectBackendPick(backendActivationRequestId, selectedAtMs)',
  "typeof rustAppShadow.clearSelection === 'function'",
  'navigation = rustAppShadow.clearSelection();',
  'interaction?.hovered != null',
  'mr_pickSurface(pickMvp, pickView, pickCameraPos, x, y)',
  'if (pick.ignored) return;',
]) {
  assert.ok(browser.includes(required), `browser pick authority is missing ${required}`);
}

assert.equal(
  browser.includes(': mirrorSelectedObjectToApp(selectedAtMs);'),
  false,
  'Rust selection still bypasses interaction authority for ordinary picks',
);

for (const retired of [
  'quiltingWasmBackend.mr_stageBackendPickEvidence(',
  'quiltingWasmBackend.mr_readBackendPickEvidence()',
  'const targetEpoch = rustInteractionDiagnostics.targetEpoch;',
  'Number(report.targetEpoch) !== currentTargetEpoch',
]) {
  assert.equal(browser.includes(retired), false, `browser retained forbidden shuttle ${retired}`);
}

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Backend pick shadow source smoke passed');
