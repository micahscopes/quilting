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
  'lodFullSceneDirty = true',
  'deviceFullScene ? 0 : (primaryOnly ? currentPrimaryFaceCount : 0)',
  'activateWebGpuLodAuthority()',
  "sameContextLodDiagnostics.effectiveAuthority = 'webgpu-device'",
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
  recompute.indexOf('if (deviceLodDispatched && deviceAuthorityEligible)')
    < recompute.indexOf('const sameContextReady = refreshSameContextLodReadiness()'),
  'a proven device epoch must exit before incumbent readiness or readback work',
);
assert.ok(recompute.includes('deviceFullScene\n            ? authoredLodStates'),
  'the visible device authority must classify the complete composed scene');
assert.ok(recompute.includes('if (wt.full_snapshot && !webGpuLodAuthorityDiagnostics.active)'),
  'worker rollback must acknowledge a fresh full snapshot');

console.log('WebGPU device-resident LOD authority source smoke passed');
