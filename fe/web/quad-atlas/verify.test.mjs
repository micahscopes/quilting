import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyQuadSnapshot } from './verify.mjs';
import { readFileSync } from 'node:fs';

test('recorded Chromium GPU snapshots satisfy planar invariants', () => {
  const snapshots = JSON.parse(readFileSync(new URL('./browser-snapshots.json', import.meta.url), 'utf8'));
  assert.equal(snapshots.cases.length, 4);
  for (const snapshot of snapshots.cases) verifyQuadSnapshot(snapshot);
});

const square = () => ({ key: [0, 0, 0, 0],
  points: [[0, 0], [16384, 0], [16384, 16384], [0, 16384]],
  triangles: [[0, 1, 2], [0, 2, 3]] });
test('parallel insertion and local relocation preserve checked planar meshes', () => {
  const evidence = JSON.parse(readFileSync(new URL('./parallel-browser-snapshots.json', import.meta.url), 'utf8'));
  assert.equal(evidence.passes, 18);
  assert.equal(evidence.cases.length, 4);
  for (const snapshot of evidence.cases) verifyQuadSnapshot(snapshot);
});
test('both cocircular square diagonals are valid', () => {
  verifyQuadSnapshot(square());
  verifyQuadSnapshot({ ...square(), triangles: [[0, 1, 3], [1, 2, 3]] });
});
test('a square with a center fan is Delaunay', () => {
  verifyQuadSnapshot({ ...square(), points: [...square().points, [8192, 8192]],
    triangles: [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]] });
});
test('missing, reversed, and overlapped triangles are rejected', () => {
  assert.throws(() => verifyQuadSnapshot({ ...square(), triangles: [[0, 1, 2]] }));
  assert.throws(() => verifyQuadSnapshot({ ...square(), triangles: [[0, 2, 1], [0, 2, 3]] }));
  assert.throws(() => verifyQuadSnapshot({ ...square(), triangles: [[0, 1, 2], [0, 1, 2]] }));
});
test('incorrect boundary and stale success receipts are rejected', () => {
  assert.throws(() => verifyQuadSnapshot({ ...square(), key: [1, 0, 0, 0] }));
  assert.throws(() => verifyQuadSnapshot({ ...square(), receipt: Array(16).fill(0) }));
});
