import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('saved Fe atomic lifecycle output and Chromium observations are coherent', () => {
  // Receipt consistency only. Running this test does not rerun the browser.
  const evidence = JSON.parse(readFileSync(new URL('./atomic-actor.browser.json', import.meta.url), 'utf8'));
  assert.match(evidence.feCommit, /^[a-f0-9]{40}$/);
  assert.match(evidence.sonatinaCommit, /^[a-f0-9]{40}$/);
  const { fixture, browser } = evidence;
  assert.deepEqual(fixture.passes.map(p => p.declaration.source_entry), ['initialize', 'claim', 'observe', 'paint']);
  assert.deepEqual(fixture.resources.map(r => r.name), ['claims', 'receipts']);
  assert.equal(fixture.resources[0].element, 'AtomicU32');
  assert.equal(fixture.resources[0].policy.access, 'atomic_read_write');
  assert.ok(fixture.resources.every(resource => !resource.artifact));
  assert.deepEqual(fixture.passes.map(p => Buffer.byteLength(p.wgsl)), browser.shaderBytes);
  for (const [index, operation, count] of [[0, 'atomicStore', 2], [1, 'atomicAdd', 2], [1, 'atomicMin', 1], [2, 'atomicLoad', 3]]) {
    assert.equal(fixture.passes[index].wgsl.split(`${operation}(`).length - 1, count);
  }
  assert.ok(!fixture.passes[3].wgsl.includes('atomic<u32>'));
  assert.equal(browser.sameDevice, true);
  assert.equal(browser.observations.length, 5);
  for (const [generation, observation] of browser.observations.entries()) {
    assert.equal(observation.generation, generation);
    assert.deepEqual(observation.claims, [128, 0, 0, 0]);
    assert.deepEqual(observation.receipts, Array(64).fill(128));
  }
});
