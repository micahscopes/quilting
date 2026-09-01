import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const core = read('crates/quilting-core/src/render.rs');
const webGpu = read('crates/quilting-webgpu/src/lib.rs');
const native = read('crates/quilting-webgpu/tests/native_lod.rs');
const bridge = read('crates/quilting-wasm/src/webgpu_backend.rs');
const renderer = read('crates/quilting-wasm/src/main_renderer.rs');
const renderShadow = read('crates/quilting-wasm/src/render_shadow.rs');

for (const required of [
  'pub struct ValidatedRenderScene {',
  'pub fn shares_snapshot_with(&self, other: &Self) -> bool',
  'pub fn upload_validated_patch_render_scene(',
  'pub fn update_validated_patch_render_scene_in_place(',
  'ShapeChanged(ValidatedRenderScene)',
]) {
  assert.ok(core.includes(required) || webGpu.includes(required),
    `shared render-scene contract is missing ${required}`);
}

for (const required of [
  'scene: ValidatedRenderScene,',
  '.update_validated_patch_render_scene_in_place(',
  '.upload_validated_patch_render_scene(',
  'resident_root_render_domains(scene.snapshot()',
]) {
  assert.ok(bridge.includes(required), `WebGPU bridge is missing ${required}`);
}

const replaceStart = bridge.indexOf('pub(crate) fn replace_scene(');
const replaceEnd = bridge.indexOf('\n/// Execute one current live frame', replaceStart);
const replaceScene = bridge.slice(replaceStart, replaceEnd);
assert.ok(replaceStart >= 0 && replaceEnd > replaceStart,
  'could not locate WebGPU scene publication');
assert.equal(replaceScene.includes('scene.revision = next_revision'), false,
  'WebGPU publication must not rewrite the shared semantic scene revision');
assert.equal(replaceScene.includes('ValidatedRenderScene::new('), false,
  'WebGPU publication must not revalidate the shared scene');

for (const required of [
  'render_scene_dirty: bool',
  'validated_render_scene: Option<ValidatedRenderScene>',
  'render_command_plan: Option<RenderCommandPlan>',
  'fn refresh_validated_render_scene(',
  'fn refresh_render_command_plan(renderer: &mut MainState, backend_plan_required: bool)',
  'fn current_render_frame(',
  'RenderFrame::from_command_plan(',
  'fn webgpu_render_style_requested(',
  'fn webgpu_frame_requested(renderer: &MainState) -> bool',
  'quilting_webgpu::supports_patch_presentation_style(style)',
  'crate::webgpu_backend::live_presentation_requested()',
  'renderer.backend_evidence_requested',
  'backend_scene_required: bool',
  'renderer.render_shadow.replace_scene(scene.clone())',
  'renderer.validated_render_scene = Some(scene)',
  'crate::webgpu_backend::needs_scene(',
  'crate::webgpu_backend::frame_contract_required()',
  'scene.clone(),',
  'plan,',
  '("renderSceneExtractions", state.render_scene_extractions)',
]) {
  assert.ok(renderer.includes(required), `shared renderer extraction is missing ${required}`);
}

const submitStart = renderer.indexOf('fn submit_webgpu_frame(');
const submitEnd = renderer.indexOf('\n#[cfg(feature = "webgpu-backend")]\nfn capture_webgl_frame_evidence', submitStart);
const submit = renderer.slice(submitStart, submitEnd);
assert.ok(submit.includes('renderer.validated_render_scene.as_ref()'),
  'WebGPU submission must consume the main renderer scene epoch');
assert.equal(submit.includes('extract_render_scene(renderer)'), false,
  'WebGPU submission must not independently extract a second scene');
assert.ok(submit.includes('frame: Option<&RenderFrame>'),
  'WebGPU submission must consume the main renderer frame');

assert.ok(bridge.includes('frame: &RenderFrame'),
  'WebGPU submission must borrow the shared render frame');
assert.ok(bridge.includes('validated_scene().shares_snapshot_with(scene)'),
  'WebGPU scene residency must require exact shared allocation identity');
assert.ok(bridge.includes('.execution(scene.scene())'),
  'WebGPU must validate the shared frame against exact retained scene identity');
assert.equal(bridge.includes('command_plan: Option<RenderCommandPlan>'), false,
  'WebGPU must not retain a parallel command-plan cache');
assert.equal(bridge.includes('RenderCommandPlan::build('), false,
  'WebGPU must not independently rebuild the shared command plan');
assert.equal(bridge.includes('RenderFrame::from_command_plan('), false,
  'WebGPU must not independently construct the shared render frame');
const liveSubmitStart = bridge.indexOf('pub(crate) fn submit_frame(');
const liveSubmitEnd = bridge.indexOf(
  '\n/// Stage a one-pixel query against the latest completed prepared-patch frame.',
  liveSubmitStart,
);
const liveSubmit = bridge.slice(liveSubmitStart, liveSubmitEnd);
assert.ok(liveSubmitStart >= 0 && liveSubmitEnd > liveSubmitStart,
  'could not locate WebGPU live frame submission');
assert.ok(liveSubmit.indexOf('.execution(scene.scene())') >= 0,
  'WebGPU live submission must validate exact frame/scene provenance');
assert.ok(liveSubmit.indexOf('.execution(scene.scene())') < liveSubmit.indexOf('let unchanged ='),
  'WebGPU must validate frame provenance before retaining an unchanged surface');

for (const required of [
  'frame: &RenderFrame',
  'frame.execution_with_command_plan(plan)',
  '.observe_execution(frame.revision, execution, actual)',
]) {
  assert.ok(renderShadow.includes(required),
    `WebGL parity observer is missing shared-frame step ${required}`);
}

const evidenceStart = renderer.indexOf('pub fn mr_request_backend_frame_evidence()');
const evidenceEnd = renderer.indexOf(
  '\n#[cfg(feature = "webgpu-backend")]\n#[wasm_bindgen(js_name = "mr_compareBackendFrameEvidence")]',
  evidenceStart,
);
const evidence = renderer.slice(evidenceStart, evidenceEnd);
assert.ok(evidenceStart >= 0 && evidenceEnd > evidenceStart,
  'could not locate backend frame evidence request');
assert.ok(evidence.includes('sync_render_batches(state);'),
  'backend evidence must synchronize pending batch semantics');
assert.ok(evidence.includes('refresh_validated_render_scene(state, true)'),
  'backend evidence must refresh the shared validated scene epoch');
assert.ok(evidence.includes('scene.snapshot().revision == state.render_command_builds'),
  'backend evidence must preflight the current structural revision');
assert.equal(evidence.includes('extract_render_scene(state)'), false,
  'backend evidence must not independently extract a private scene');

for (const required of [
  'ValidatedRenderScene::new(render_scene.clone())',
  '.upload_validated_patch_render_scene(',
  '.shares_snapshot_with(&validated_scene)',
]) {
  assert.ok(native.includes(required), `native allocation-sharing oracle is missing ${required}`);
}

console.log('Shared validated render-scene extraction source smoke passed');
