import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const backend = read('crates/quilting-wasm/src/webgpu_backend.rs');
const renderer = read('crates/quilting-wasm/src/main_renderer.rs');
const renderContract = read('crates/quilting-core/src/render.rs');
const webgpu = read('crates/quilting-webgpu/src/lib.rs');
const roots = read('crates/quilting-webgpu/src/resident_roots.rs');
const overlay = read('crates/quilting-webgpu/src/adaptive_overlay.rs');

for (const required of [
  'fallback_pose_uploads: u64',
  'fallback_pose_initializations: u64',
  'fallback_pose_reuses: u64',
  'classifier_pose_uploads: u64',
  'classifier_pose_reuses: u64',
  'resident_pose_uploads: u64',
  'resident_pose_initializations: u64',
  'resident_pose_reuses: u64',
  'device_pose_identity: Option<RenderPoseIdentity>',
  'patch_pose_uniforms_ready: bool',
  'resident_pose_uniforms_ready: bool',
  'let pose_upload = if backend.device_pose_identity == Some(pose_identity)',
  'fn render_pose_upload_policy(',
  'render_pose_upload_policy(',
  'if !resident_root_frame && pose_upload_required',
  'backend.fallback_pose_uploads.saturating_add(1)',
  'backend.fallback_pose_initializations.saturating_add(1)',
  'backend.fallback_pose_reuses.saturating_add(1)',
  'backend.classifier_pose_uploads.saturating_add(1)',
  'backend.classifier_pose_reuses.saturating_add(1)',
  'PoseUploadPolicy::Publish',
  'PoseUploadPolicy::PublishPreparation',
  'PoseUploadPolicy::Reuse',
  'backend.resident_pose_uploads.saturating_add(1)',
  'backend.resident_pose_initializations.saturating_add(1)',
  'backend.resident_pose_reuses.saturating_add(1)',
]) {
  assert.ok(backend.includes(required), `pose memo is missing ${required}`);
}

for (const required of [
  'pub enum PoseUploadPolicy',
  'Publish',
  'PublishPreparation',
  'Reuse',
  'pub const fn should_publish_dynamic',
  'pub const fn should_publish_preparation',
]) {
  assert.ok(webgpu.includes(required), `typed pose policy is missing ${required}`);
}
assert.ok(
  roots.includes('if pose_upload.should_publish_dynamic() {\n            self.write_dynamic_pose'),
  'resident roots must separate shared dynamic pose publication',
);
assert.ok(
  roots.includes('if pose_upload.should_publish_preparation() {\n            self.write_patch_joint_count'),
  'resident roots must initialize preparation-local state independently',
);
assert.ok(
  overlay.includes('if write_dynamic_pose && pose_upload.should_publish_dynamic() {'),
  'adaptive overlay must share the root dynamic-pose policy',
);
assert.ok(
  overlay.includes('if pose_upload.should_publish_preparation() {'),
  'adaptive overlay must initialize its preparation state with the root family',
);

const submitStart = backend.indexOf('pub(crate) fn submit_frame(');
const submitEnd = backend.indexOf(
  '\n/// Stage a one-pixel query against the latest completed prepared-patch frame.',
  submitStart,
);
const submit = backend.slice(submitStart, submitEnd);
assert.ok(submitStart >= 0 && submitEnd > submitStart,
  'could not locate WebGPU live frame submission');
assert.ok(submit.indexOf('let pose_upload = render_pose_upload_policy(')
  < submit.indexOf('let unchanged ='),
  'semantic pose identity must participate in unchanged-frame admission');
assert.ok(submit.indexOf('if !resident_root_frame && pose_upload_required')
  < submit.indexOf('write_patch_render_pose_state('),
  'fallback pose upload must be guarded before the queue write');
assert.ok(submit.includes('pose_upload,'),
  'resident frame submission must forward its typed pose policy');
assert.ok(renderer.includes('surface_runtime.current_pose_payload()'),
  'the live renderer must still provide the exact retained pose payload');
assert.ok(renderer.includes('fn current_render_pose_identity('),
  'the main renderer must derive one semantic pose identity');
assert.ok(renderer.includes('RenderPoseIdentity::timed('),
  'timed frames must include animation continuity in their pose identity');
assert.ok(renderContract.includes('pub const fn timed('),
  'the backend-neutral render contract must own timed pose identity packing');
assert.ok(!backend.includes('last_joint_matrices'),
  'the backend must not retain a second joint-pose vector for comparisons');
assert.ok(!backend.includes('last_morph_weights'),
  'the backend must not retain a second morph-pose vector for comparisons');

console.log('WebGPU retained pose-upload memo source smoke passed');
