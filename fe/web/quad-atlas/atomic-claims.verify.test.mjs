import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('saved browser atomic prerequisite has exact contention and wraparound receipts', () => {
  const proof = JSON.parse(readFileSync(new URL('./atomic-claims.browser.json', import.meta.url)));
  assert.equal(proof.shaderBytes, Buffer.byteLength(proof.wgsl));
  assert.equal(proof.receipts.length, 10);
  assert.match(proof.wgsl, /array<atomic<u32>>/);
  assert.match(proof.wgsl, /atomicAdd\(/);
  assert.match(proof.wgsl, /atomicMin\(/);
  assert.deepEqual([...new Set(proof.receipts.map(r => r.workgroups))], [1, 2, 16, 64, 255]);
  for (const r of proof.receipts) {
    assert.equal(r.invocations, r.workgroups * 64);
    assert.equal(r.updates, r.invocations * 2);
    assert.deepEqual(r.actual, [(r.initialCount + r.updates) >>> 0, 0, 0x12345678, 0x9abcdef0]);
  }
});
