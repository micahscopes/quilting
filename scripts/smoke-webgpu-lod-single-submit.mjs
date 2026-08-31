#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const webgpu = read('crates/quilting-webgpu/src/lib.rs');
const browser = read('crates/quilting-wasm/src/webgpu_backend.rs');

const graphStart = webgpu.indexOf('pub fn classify_and_reconcile_on_device');
const graphEnd = webgpu.indexOf('pub async fn read_lod_classification_for_diagnostics', graphStart);
const graph = webgpu.slice(graphStart, graphEnd);
assert.ok(graphStart >= 0 && graphEnd > graphStart,
  'could not isolate the classified resident LOD graph');
assert.ok(graph.includes('self.write_lod_classification_state('),
  'the graph must publish or reuse classifier inputs before encoding');
assert.ok(graph.indexOf('self.encode_lod_classification(')
  < graph.indexOf('self.encode_resident_lod_reconciliation('),
  'classification must precede reconciliation in one encoder');
assert.equal(graph.match(/create_command_encoder/g)?.length, 1,
  'the complete LOD graph must allocate one command encoder');
assert.equal(graph.match(/self\.queue\.submit/g)?.length, 1,
  'the complete LOD graph must issue one queue submission');
assert.ok(!graph.includes('readback') && !graph.includes('copy_buffer_to_buffer'),
  'the production graph must remain device-local');

assert.ok(browser.includes('.classify_and_reconcile_on_device('),
  'the browser must consume the single-submit LOD graph');
assert.ok(!browser.includes('.classify_on_device(')
  && !browser.includes('.reconcile_resident_lod_on_device('),
  'the browser must not retain the former two-submit sequence');

console.log('WebGPU single-submit resident LOD graph source smoke passed');
