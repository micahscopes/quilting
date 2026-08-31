#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const webgpu = read('crates/quilting-webgpu/src/lib.rs');
const prepared = read('crates/quilting-webgpu/src/prepared_patch_pipeline.rs');
const roots = read('crates/quilting-webgpu/src/resident_roots.rs');
const overlay = read('crates/quilting-webgpu/src/adaptive_overlay.rs');
const patchShader = read('crates/quilting-shaders/shaders/render/patch.wgsl');
const rootShader = read('crates/quilting-shaders/shaders/render/resident_root_patch.wgsl');
const patchPick = read('crates/quilting-shaders/shaders/render/patch_pick.wgsl');
const rootPick = read('crates/quilting-shaders/shaders/render/resident_root_pick.wgsl');
const patchVisibility = read('crates/quilting-shaders/shaders/compute/prepared_visibility.wgsl');
const rootVisibility = read('crates/quilting-shaders/shaders/compute/resident_visibility.wgsl');

for (const required of [
  'const PATCH_RENDER_GLOBAL_WORDS: usize = 44;',
  'const PATCH_RENDER_GLOBAL_BYTES: u64 = 176;',
  'const PATCH_RENDER_DOMAIN_WORDS: usize = 20;',
  'const PATCH_RENDER_DOMAIN_BYTES: u64 = 80;',
  'pub struct PatchRenderGlobal',
  'pub struct PatchRenderDomain',
  'global_frame: Arc<PatchRenderGlobalResidency>',
  'domains: wgpu::Buffer',
  'PatchRenderGlobal::from_render_frame(frame, use_qb)',
  'PatchRenderDomain::from_transform(',
]) {
  assert.ok(webgpu.includes(required), `split frame contract is missing ${required}`);
}
assert.ok(!webgpu.includes('const PATCH_RENDER_FRAME_BYTES'),
  'the monolithic 256-byte device row must be retired');
assert.ok(prepared.includes('PATCH_RENDER_GLOBAL_BYTES')
  && prepared.includes('PATCH_RENDER_DOMAIN_BYTES'),
  'prepared functional layouts must name both split records');
assert.ok(roots.includes('global_frame: Arc<PatchRenderGlobalResidency>')
  && roots.includes('render_domains: wgpu::Buffer'),
  'resident roots must retain the split buffers');
assert.ok(overlay.includes('adaptive patch domain rows')
  && overlay.includes('material_slot'),
  'adaptive visibility and material domains must use local rows');

for (const [name, source] of [
  ['prepared render', patchShader],
  ['resident render', rootShader],
  ['prepared picking', patchPick],
  ['resident picking', rootPick],
  ['prepared visibility', patchVisibility],
  ['resident visibility', rootVisibility],
]) {
  assert.ok(source.includes('PatchRenderGlobal'), `${name} lacks global frame state`);
  assert.ok(source.includes('PatchRenderDomain'), `${name} lacks local domain state`);
  assert.ok(!source.includes('PatchRenderFrame'), `${name} retained the monolithic record`);
}

console.log('WebGPU split frame-state source smoke passed');
