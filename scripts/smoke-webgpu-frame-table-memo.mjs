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
const backend = read('crates/quilting-wasm/src/webgpu_backend.rs');

for (const required of [
  'struct RetainedFrameTable',
  'published: bool',
  'fn begin_update(&self) -> bool',
  'fn replace_row(',
  'fn invalidate(&mut self)',
  'fn commit(&mut self, changed: bool) -> FrameTablePublication',
  'FrameTablePublication::Reuse',
  'frame_table_uploads: AtomicU64',
  'frame_table_reuses: AtomicU64',
  'frame_table_upload_bytes: AtomicU64',
  'pub fn frame_table_memo_diagnostics',
  'self.write_patch_render_frame_parts(',
  'global_frame_table: Mutex<RetainedFrameTable>',
  'domain_table: Mutex<RetainedFrameTable>',
]) {
  assert.ok(webgpu.includes(required), `frame-table memo is missing ${required}`);
}

const frameEncoderStart = webgpu.indexOf('fn encode_render_frame_with_pipelines');
const frameEncoderEnd = webgpu.indexOf('pub fn encode_diagnostic_render_frame', frameEncoderStart);
const frameEncoder = webgpu.slice(frameEncoderStart, frameEncoderEnd);
assert.ok(frameEncoderStart >= 0 && frameEncoderEnd > frameEncoderStart,
  'could not isolate retained frame encoding');
assert.ok(!frameEncoder.includes('collect::<Result<Vec<_>, LodWebGpuError>>()?'),
  'production fallback encoding must not allocate a temporary frame vector');
assert.ok(frameEncoder.includes('self.write_patch_render_frame_parts('),
  'fallback batches must pack directly into retained staging words');

assert.ok(roots.includes('let mut global_changed = global_table.begin_update();'),
  'resident roots must memoize their global frame');
assert.ok(roots.includes('let mut domains_changed = domain_table.begin_update();'),
  'resident roots must memoize their local domains');
assert.ok(roots.includes('domain_table.invalidate();'),
  'resident roots must invalidate partial local staging after failure');
assert.ok(roots.includes('self.record_frame_table_publication(global_publication);')
  && roots.includes('self.record_frame_table_publication(domain_publication);'),
  'resident roots must expose both publications');
assert.ok(overlay.includes('self.write_patch_render_frame_parts('),
  'adaptive overlays must use the shared retained split-frame memo');

for (const required of [
  'frame_table_uploads: u64',
  'frame_table_reuses: u64',
  'frame_table_upload_bytes: u64',
  'LodClassifierDevice::frame_table_memo_diagnostics',
]) {
  assert.ok(backend.includes(required), `browser diagnostics are missing ${required}`);
}

console.log('WebGPU retained frame-table memo source smoke passed');
