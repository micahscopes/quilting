#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const renderer = read('crates/quilting-renderer/src/compute.rs');
const webgpu = read('crates/quilting-webgpu/src/lib.rs');
const browser = read('crates/quilting-wasm/src/webgpu_backend.rs');

for (const required of [
  'pub fn pack_wgsl_lod_subject_words_into',
  'packed.clear();',
  'packed.resize(subject_rows.len().max(1), [0; 40]);',
]) {
  assert.ok(renderer.includes(required), `reusable LOD subject packer is missing ${required}`);
}

for (const required of [
  'struct RetainedLodDispatchState',
  'subject_scratch: Vec<[u32; 40]>',
  'uniform_published: bool',
  'subjects_published: bool',
  'fn commit_uniform',
  'fn commit_subject_scratch',
  'lod_state: Mutex<RetainedLodDispatchState>',
  'pack_wgsl_lod_subject_words_into(',
  'self.record_lod_state_publication(uniform_publication);',
  'self.record_lod_state_publication(subject_publication);',
  'pub fn lod_state_memo_diagnostics',
]) {
  assert.ok(webgpu.includes(required), `retained LOD state is missing ${required}`);
}

const writerStart = webgpu.indexOf('pub fn write_lod_classification_state');
const writerEnd = webgpu.indexOf('pub fn encode_lod_classification', writerStart);
const writer = webgpu.slice(writerStart, writerEnd);
assert.ok(writerStart >= 0 && writerEnd > writerStart, 'could not isolate LOD state writer');
assert.ok(writer.includes('matches!(uniform_publication, FrameTablePublication::Upload'),
  'LOD uniform queue writes must require an exact publication miss');
assert.ok(writer.includes('matches!(subject_publication, FrameTablePublication::Upload'),
  'LOD subject queue writes must require an exact publication miss');
assert.ok(!writer.includes('pack_wgsl_lod_subject_words(&'),
  'production LOD state must not allocate the legacy returned subject vector');

for (const required of [
  'lod_state_uploads: u64',
  'lod_state_reuses: u64',
  'lod_state_upload_bytes: u64',
  'LodClassifierDevice::lod_state_memo_diagnostics',
]) {
  assert.ok(browser.includes(required), `browser LOD diagnostics are missing ${required}`);
}

console.log('WebGPU retained LOD-state memo source smoke passed');
