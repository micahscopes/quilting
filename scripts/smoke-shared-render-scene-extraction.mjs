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
  'fn refresh_validated_render_scene(',
  'backend_scene_required: bool',
  'renderer.render_shadow.replace_scene(scene.clone())',
  'renderer.validated_render_scene = Some(scene)',
  'crate::webgpu_backend::needs_scene(',
  'scene.clone(),',
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

for (const required of [
  'ValidatedRenderScene::new(render_scene.clone())',
  '.upload_validated_patch_render_scene(',
  '.shares_snapshot_with(&validated_scene)',
]) {
  assert.ok(native.includes(required), `native allocation-sharing oracle is missing ${required}`);
}

console.log('Shared validated render-scene extraction source smoke passed');
