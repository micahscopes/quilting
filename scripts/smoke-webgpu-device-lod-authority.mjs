#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const source = readFileSync(join(repository, 'hyperscope.html'), 'utf8');

for (const required of [
  'globalThis.__hyperscopeWebGpuLodAuthority = webGpuLodAuthorityDiagnostics',
  "previousEffective === 'webgpu' && !presenting",
  "retireWebGpuLodAuthority('WebGPU presentation retired')",
  'function webGpuLodAuthorityEligible()',
  "graphicsBackendDiagnostics.effective === 'webgpu'",
  "graphicsBackendDiagnostics.state === 'presenting'",
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
]) {
  assert.ok(source.includes(required), `device LOD authority is missing ${required}`);
}

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
  'the visible device authority must classify the complete composed scene');
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
