#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const webgpu = read('crates/quilting-webgpu/src/lib.rs');
const roots = read('crates/quilting-webgpu/src/resident_roots.rs');
const overlay = read('crates/quilting-webgpu/src/adaptive_overlay.rs');
const browser = read('crates/quilting-wasm/src/webgpu_backend.rs');

for (const required of [
  'struct PatchRenderGlobalResidency',
  'buffer: wgpu::Buffer',
  'table: Mutex<RetainedFrameTable>',
  'global_frame: Arc<PatchRenderGlobalResidency>',
  'shared_global_frame: Option<Arc<PatchRenderGlobalResidency>>',
  'self.write_patch_render_global(&bindings.global_frame, global)?;',
]) {
  assert.ok(webgpu.includes(required), `aggregate global residency is missing ${required}`);
}
assert.ok(!webgpu.includes('global_frame_table: Mutex<RetainedFrameTable>'),
  'ordinary render bindings must not retain a second global memo outside shared residency');
assert.ok(roots.includes('global_frame: Arc<PatchRenderGlobalResidency>'),
  'resident roots must own the aggregate-global residency');
assert.ok(!roots.includes('global_frame_table: Mutex<RetainedFrameTable>'),
  'resident roots must not retain a parallel global memo');

for (const required of [
  'upload_adaptive_overlay_scene_with_pbr_resources_for_roots',
  'upload_focus_adaptive_overlay_scene_with_pbr_resources_for_roots',
  'Some(root_bindings)',
  'Arc::clone(&bindings.global_frame)',
  'pub fn shares_global_frame_with',
  'Arc::ptr_eq(&self.bindings.global_frame, &roots.global_frame)',
  'bindings.domain_identity != roots.draw_domains.domain_identity',
]) {
  assert.ok(overlay.includes(required), `root-scoped overlay construction is missing ${required}`);
}

assert.ok(browser.includes('.upload_adaptive_overlay_scene_with_pbr_resources_for_roots('),
  'the browser diagnostic aggregate must use root-scoped global residency');
assert.ok(browser.includes('.upload_focus_adaptive_overlay_scene_with_pbr_resources_for_roots('),
  'the browser focus aggregate must use root-scoped global residency');
assert.ok(browser.match(/\.shares_global_frame_with\(/g)?.length === 2,
  'both diagnostic and focus browser aggregates must prove shared residency');

console.log('WebGPU aggregate-global frame residency source smoke passed');
