import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const backend = read('crates/quilting-wasm/src/webgpu_backend.rs');
const renderer = read('crates/quilting-wasm/src/main_renderer.rs');
const webgpu = read('crates/quilting-webgpu/src/lib.rs');
const roots = read('crates/quilting-webgpu/src/resident_roots.rs');
const overlay = read('crates/quilting-webgpu/src/adaptive_overlay.rs');

for (const required of [
  'fallback_pose_uploads: u64',
  'fallback_pose_reuses: u64',
  'resident_pose_uploads: u64',
  'resident_pose_reuses: u64',
  'let pose_upload_required = backend.last_frame_input.is_none()',
  '|| backend.last_joint_matrices != joint_matrices',
  '|| backend.last_morph_weights != effective_morph_weights',
  'if !resident_root_frame && pose_upload_required',
  'backend.fallback_pose_uploads.saturating_add(1)',
  'backend.fallback_pose_reuses.saturating_add(1)',
  'PoseUploadPolicy::Publish',
  'PoseUploadPolicy::Reuse',
  'backend.resident_pose_uploads.saturating_add(1)',
  'backend.resident_pose_reuses.saturating_add(1)',
]) {
  assert.ok(backend.includes(required), `pose memo is missing ${required}`);
}

for (const required of [
  'pub enum PoseUploadPolicy',
  'Publish',
  'Reuse',
  'const fn should_publish',
]) {
  assert.ok(webgpu.includes(required), `typed pose policy is missing ${required}`);
}
assert.ok(
  roots.includes('if pose_upload.should_publish() {\n            self.write_resident_root_preparation_pose'),
  'resident root pose publication must obey the typed policy',
);
assert.ok(
  overlay.includes('if publish_pose_state {\n            if write_dynamic_pose {'),
  'adaptive overlay pose publication must be atomic with the root policy',
);

const submitStart = backend.indexOf('pub(crate) fn submit_frame(');
const submitEnd = backend.indexOf(
  '\n/// Stage a one-pixel query against the latest completed prepared-patch frame.',
  submitStart,
);
const submit = backend.slice(submitStart, submitEnd);
assert.ok(submitStart >= 0 && submitEnd > submitStart,
  'could not locate WebGPU live frame submission');
assert.ok(submit.indexOf('let pose_upload_required =') < submit.indexOf('let unchanged ='),
  'pose identity must participate in unchanged-frame admission');
assert.ok(submit.indexOf('if !resident_root_frame && pose_upload_required')
  < submit.indexOf('write_patch_render_pose_state('),
  'fallback pose upload must be guarded before the queue write');
assert.ok(submit.includes('pose_upload,'),
  'resident frame submission must forward its typed pose policy');
assert.ok(renderer.includes('surface_runtime.current_pose_payload()'),
  'the live renderer must still provide the exact retained pose payload');

console.log('WebGPU retained pose-upload memo source smoke passed');
