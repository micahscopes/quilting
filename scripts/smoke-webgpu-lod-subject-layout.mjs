#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const renderer = read('crates/quilting-renderer/src/compute.rs');
const webgpu = read('crates/quilting-webgpu/src/lib.rs');

for (const required of [
  'pub struct WgslLodSubjectLayout',
  'pub fn from_prepared(prepared: &PreparedLodModel)',
  'pub subject_layout: WgslLodSubjectLayout',
  'pub fn pack_wgsl_lod_subject_words_with_layout',
  'pack_wgsl_lod_subject_words_with_layout(&layout, dispatch, packed)',
]) {
  assert.ok(renderer.includes(required), `retained subject layout is missing ${required}`);
}

for (const required of [
  'subject_layout: WgslLodSubjectLayout',
  'let subject_rows = words.subject_layout.len().max(1);',
  'subject_layout: words.subject_layout',
  'pack_wgsl_lod_subject_words_with_layout(',
  '&model.subject_layout',
]) {
  assert.ok(webgpu.includes(required), `uploaded-model subject layout is missing ${required}`);
}

const writerStart = webgpu.indexOf('pub fn write_lod_classification_state');
const writerEnd = webgpu.indexOf('pub fn encode_lod_classification', writerStart);
const writer = webgpu.slice(writerStart, writerEnd);
assert.ok(!writer.includes('WgslLodSubjectLayout::from_prepared')
  && !writer.includes('pack_wgsl_lod_subject_words_into('),
  'production classification must not reconstruct the subject map');

console.log('WebGPU retained LOD subject-layout source smoke passed');
