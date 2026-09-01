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
const graphEnd = webgpu.indexOf('pub fn refresh_resident_lod_prefix_on_device', graphStart);
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

const prefixStart = webgpu.indexOf('pub fn refresh_resident_lod_prefix_on_device');
const prefixEnd = webgpu.indexOf(
  'pub async fn read_lod_classification_for_diagnostics',
  prefixStart,
);
const prefixGraph = webgpu.slice(prefixStart, prefixEnd);
assert.ok(prefixStart >= 0 && prefixEnd > prefixStart,
  'could not isolate the prefix-refresh resident LOD graph');
assert.ok(prefixGraph.includes('validate_lod_classification_prefix('),
  'a partial graph must prove that its prefix is topology-closed');
assert.ok(prefixGraph.includes('partial LOD refresh requires a complete resident baseline'),
  'a partial graph must never manufacture an incomplete initial epoch');
assert.ok(prefixGraph.indexOf('self.encode_lod_classification_prefix(')
  < prefixGraph.indexOf('self.encode_resident_lod_reconciliation('),
  'prefix classification must precede prefix reconciliation in one encoder');
assert.equal(prefixGraph.match(/create_command_encoder/g)?.length, 1,
  'the prefix LOD graph must allocate one command encoder');
assert.equal(prefixGraph.match(/self\.queue\.submit/g)?.length, 1,
  'the prefix LOD graph must issue one queue submission');
assert.ok(!prefixGraph.includes('readback') && !prefixGraph.includes('copy_buffer_to_buffer'),
  'the prefix graph must retain its suffix without CPU traffic');

assert.ok(browser.includes('.classify_and_reconcile_on_device('),
  'the browser must consume the single-submit LOD graph');
assert.ok(browser.includes('.refresh_resident_lod_prefix_on_device('),
  'the browser must consume the topology-closed prefix graph');
for (const counter of [
  'device_lod_full_dispatches',
  'device_lod_prefix_dispatches',
  'last_device_lod_classified_faces',
]) {
  assert.ok(browser.includes(counter), `browser diagnostics are missing ${counter}`);
}
assert.ok(!browser.includes('.classify_on_device(')
  && !browser.includes('.reconcile_resident_lod_on_device('),
  'the browser must not retain the former two-submit sequence');

console.log('WebGPU single-submit resident LOD graph source smoke passed');
