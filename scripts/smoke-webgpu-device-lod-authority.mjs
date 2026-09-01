#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const source = readFileSync(join(repository, 'hyperscope.html'), 'utf8');
const appAuthority = readFileSync(
  join(repository, 'crates/hyperscope-app/src/lod_authority.rs'),
  'utf8',
);
const wasmBackend = readFileSync(
  join(repository, 'crates/quilting-wasm/src/webgpu_backend.rs'),
  'utf8',
);

for (const required of [
  'globalThis.__hyperscopeWebGpuLodAuthority = webGpuLodAuthorityDiagnostics',
  "previousEffective === 'webgpu' && !presenting",
  "retireWebGpuLodAuthority('WebGPU presentation retired')",
  'function webGpuLodAuthorityEligible()',
  'graphicsPresentationPolicyDiagnostics.lastEffective',
  '.deviceLodAuthorityEligible',
  'function activateWebGpuLodAuthority()',
  'lodDeltaResetPending = true',
  'function retireWebGpuLodAuthority(reason)',
  'function requireIncumbentLodRecovery()',
  'function completeIncumbentLodRecovery(fullSnapshot, context)',
  'lodFullSceneDirty = true',
  'deviceFullScene ? 0 : (primaryOnly ? currentPrimaryFaceCount : 0)',
  'activateWebGpuLodAuthority()',
  "sameContextLodDiagnostics.effectiveAuthority = 'webgpu-device'",
  "callRustWebGpuLodAuthority(\n          'beginWebGpuLodDispatch'",
  "callRustWebGpuLodAuthority(\n            'completeWebGpuLodDispatch'",
  'chooseWebGpuLodAuthorityFlag(',
  'WEBGPU_LOD_COMPLETE_SCENE',
  'WEBGPU_LOD_DEVICE_AUTHORITY',
  'WEBGPU_LOD_INCUMBENT_REQUIRED',
  'recordWebGpuLodAuthorityParity(',
  'webGpuLodAuthorityDiagnostics.mismatches.length > 16',
  'const deviceCompleteSceneRequired = deviceRecoveryEligible',
  '(!webGpuLodAuthorityDiagnostics.active || !primaryOnly)',
]) {
  assert.ok(source.includes(required), `device LOD authority is missing ${required}`);
}

assert.ok(appAuthority.includes('WebGpuLodAuthorityReason::DevicePrefixAccepted'),
  'Rust authority must distinguish an accepted prefix refresh');
assert.ok(appAuthority.includes('complete_scene_required\n                || (self.presentation_authoritative && !self.active)'),
  'Rust authority must require a complete epoch only for activation or an explicit dirty scene');
assert.ok(wasmBackend.includes('.refresh_resident_lod_prefix_on_device('),
  'the WASM backend must keep a device-resident suffix for primary-prefix refreshes');
assert.ok(wasmBackend.includes('device_lod_prefix_dispatches'),
  'the WASM diagnostics must expose measured prefix dispatches');

const recomputeStart = source.indexOf('async function recomputeLods()');
const recomputeEnd = source.indexOf('\nasync function loadModel(', recomputeStart);
const recompute = source.slice(recomputeStart, recomputeEnd);
assert.ok(recomputeStart >= 0 && recomputeEnd > recomputeStart,
  'could not isolate LOD recomputation');
assert.ok(
  recompute.indexOf('mr_dispatchWebGpuLod(') < recompute.indexOf('mr_dispatchSameContextLod('),
  'the device authority decision must precede incumbent dispatch',
);
assert.ok(
  recompute.indexOf('WEBGPU_LOD_DEVICE_AUTHORITY,')
    < recompute.indexOf('const sameContextReady = refreshSameContextLodReadiness()'),
  'the selected device authority must exit before incumbent readiness or readback work',
);
assert.ok(recompute.includes('deviceFullScene\n            ? authoredLodStates'),
  'a complete device epoch must classify the composed scene');
assert.ok(recompute.includes(': (primaryOnly ? presentationComposition.primaryLodStates : authoredLodStates)'),
  'an active animation-only epoch must upload only the primary subject prefix');
assert.ok(recompute.includes('deviceFullScene ? 0 : (primaryOnly ? currentPrimaryFaceCount : 0)'),
  'the device dispatch must bound an animation-only classification to primary faces');
assert.ok(recompute.includes('if (deviceFullScene) {\n          webGpuLodAuthorityDiagnostics.fullSceneDispatches += 1;'),
  'full-scene diagnostics must count the resolved dispatch scope, not mere eligibility');
assert.ok(recompute.includes("completeIncumbentLodRecovery(\n          !!wt.full_snapshot,\n          'worker-publication'"),
  'worker rollback must report whether its publication is a full snapshot');
assert.ok(
  recompute.indexOf("'beginWebGpuLodDispatch'")
    < recompute.indexOf('mr_dispatchWebGpuLod(')
    && recompute.indexOf('mr_dispatchWebGpuLod(')
      < recompute.indexOf("'completeWebGpuLodDispatch'"),
  'Rust must bracket the synchronous device dispatch with begin/completion evidence',
);

console.log('WebGPU device-resident LOD authority source smoke passed');
