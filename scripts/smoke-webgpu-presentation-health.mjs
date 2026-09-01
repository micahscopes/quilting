import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const backend = read('crates/quilting-wasm/src/webgpu_backend.rs');
const appShadow = read('crates/quilting-wasm/src/app_shadow.rs');
const lodAuthority = read('crates/hyperscope-app/src/lod_authority.rs');
const browser = read('hyperscope.html');

for (const required of [
  'last_presentation_input: Option<LiveFrameInput>',
  'presentation_frame_admitted: bool',
  'backend.last_presentation_input == Some(frame_input)',
  'backend.last_presentation_input = Some(frame_input);',
  'backend.presentation_frame_admitted = true;',
  'backend.presentation_frame_admitted = false;',
  'backend.last_frame_failure = None;',
]) {
  assert.ok(backend.includes(required), `WebGPU presentation health is missing ${required}`);
}
for (const required of [
  'function refreshWebGpuBackendDiagnostics(armRenderedFrame = false)',
  'if (armRenderedFrame) {',
  'refreshWebGpuBackendDiagnostics(true);',
]) {
  assert.ok(browser.includes(required), `post-render readiness is missing ${required}`);
}
assert.equal(
  backend.includes('last_frame_presented'),
  false,
  'target-ambiguous presentation boolean survived',
);

for (const required of [
  'complete_scene_required: bool',
  'complete_scene: complete_scene_required',
  '|| (self.presentation_authoritative && !self.active),',
  'complete_recovery_dispatch_prewarms_without_seizing_authority',
]) {
  assert.ok(lodAuthority.includes(required), `Rust LOD recovery policy is missing ${required}`);
}
for (const required of [
  'presentation_authoritative: bool,\n        complete_scene_required: bool,',
  '.begin_dispatch(complete_scene_required)',
]) {
  assert.ok(appShadow.includes(required), `AppShadow LOD recovery boundary is missing ${required}`);
}

assert.ok(
  browser.includes('residency?.presentationFrameAdmitted === true'),
  'browser does not require the current Rust admission witness',
);
for (const required of [
  'function webGpuLodRecoveryEligible()',
  'const deviceRecoveryEligible = webGpuLodRecoveryEligible();',
  'const deviceCompleteSceneRequired = deviceRecoveryEligible',
  "'beginWebGpuLodDispatch',",
  'deviceAuthorityEligible,\n          deviceCompleteSceneRequired,',
  '(deviceCompleteSceneRequired ? WEBGPU_LOD_COMPLETE_SCENE : 0)',
]) {
  assert.ok(browser.includes(required), `WebGPU recovery dispatch is missing ${required}`);
}
assert.equal(
  browser.includes('(residency?.frameFailures || 0) > 0'),
  false,
  'browser still treats the cumulative frame-failure counter as current health',
);
assert.ok(
  browser.includes('(residency?.presentationLosses || 0) > 0'),
  'unrecoverable surface loss no longer retires presentation',
);

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('WebGPU presentation health source smoke passed');
