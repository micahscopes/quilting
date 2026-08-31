#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const focus = read('crates/quilting-webgpu/src/focus_postprocess.rs');
const exports = read('crates/quilting-webgpu/src/lib.rs');
const native = read('crates/quilting-webgpu/tests/native_lod.rs');
const browser = read('crates/quilting-wasm/src/webgpu_backend.rs');

for (const required of [
  'struct FocusPostprocessPlanKey',
  'blur_strength_bits: u32',
  'focus_coordinate_bits: u32',
  'stretch_range_bits: [u32; 2]',
  'struct FocusEncodingScratch',
  'plan_key: Option<FocusPostprocessPlanKey>',
  'fn prepare(',
  'if self.plan_key == Some(key)',
  'FocusPlanPublication::Reuse',
  'pub struct FocusPostprocessMemoDiagnostics',
  'pub fn memo_diagnostics(&self)',
  'plan_reused: matches!(publication, FocusPlanPublication::Reuse)',
]) {
  assert.ok(focus.includes(required), `retained focus plan is missing ${required}`);
}

const encoderStart = focus.indexOf('pub fn encode_focus_postprocess');
const encoderEnd = focus.indexOf('\nfn sampled_texture_layout', encoderStart);
const encoder = focus.slice(encoderStart, encoderEnd);
assert.ok(encoderStart >= 0 && encoderEnd > encoderStart,
  'could not isolate the focus encoder');
assert.ok(encoder.includes('FocusPlanPublication::Upload { bytes } =>'),
  'focus uniform publication must be conditional on a plan miss');
assert.ok(encoder.includes('FocusPlanPublication::Reuse => 0'),
  'a retained focus plan must report zero queue-upload bytes');
assert.equal((encoder.match(/\.write_buffer\(/g) ?? []).length, 1,
  'the focus encoder must retain one bounded conditional queue write');

assert.ok(exports.includes('FocusPostprocessMemoDiagnostics'),
  'focus memo diagnostics are not exported');
assert.ok(native.includes('assert!(root_focus_encoding.postprocess.plan_reused);'),
  'native root/adaptive reuse evidence is missing');
assert.ok(native.includes('assert_eq!(root_focus_encoding.postprocess.uniform_upload_bytes, 0);'),
  'native zero-upload evidence is missing');
for (const required of [
  'focus_plan_builds: u64',
  'focus_plan_reuses: u64',
  'focus_uniform_uploads: u64',
  'focus_uniform_reuses: u64',
  'focus_uniform_upload_bytes: u64',
  'target.memo_diagnostics().ok()',
]) {
  assert.ok(browser.includes(required), `browser focus diagnostics are missing ${required}`);
}

console.log('WebGPU retained focus-plan memo source smoke passed');
