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
  const baseline = JSON.parse(readFileSync(new URL('./browser-snapshots.json', import.meta.url), 'utf8'));
  const topology = triangles => triangles.map(t => [...t].sort((a,b) => a-b).join(',')).sort();
  for (const [index, snapshot] of evidence.cases.entries()) {
    assert.equal(snapshot.passCount, evidence.passes);
    assert.equal(new URL(snapshot.manifest).pathname, `/assets/${evidence.manifest}`);
    verifyQuadSnapshot(snapshot);
    assert.deepEqual(snapshot.points, baseline.cases[index].points);
    assert.deepEqual(topology(snapshot.triangles), topology(baseline.cases[index].triangles));
  }
});
test('both cocircular square diagonals are valid', () => {
  verifyQuadSnapshot(square());
  verifyQuadSnapshot({ ...square(), triangles: [[0, 1, 3], [1, 2, 3]] });
});

test('atomic claims preserve baseline points and final Delaunay topology', () => {
  const evidence = JSON.parse(readFileSync(new URL('./claims-browser-snapshots.json', import.meta.url), 'utf8'));
  const baseline = JSON.parse(readFileSync(new URL('./browser-snapshots.json', import.meta.url), 'utf8'));
  const topology = triangles => triangles.map(t => [...t].sort((a,b) => a-b).join(',')).sort();
  assert.equal(evidence.passes, 20);
  assert.equal(evidence.cases.length, baseline.cases.length);
  for (const [i, snapshot] of evidence.cases.entries()) {
    assert.equal(snapshot.passCount, evidence.passes);
    assert.equal(new URL(snapshot.manifest).pathname, `/assets/${evidence.manifest}`);
    verifyQuadSnapshot(snapshot);
    assert.deepEqual(snapshot.key, baseline.cases[i].key);
    assert.deepEqual(snapshot.points, baseline.cases[i].points);
    assert.deepEqual(topology(snapshot.triangles), topology(baseline.cases[i].triangles));
  }
});

test('recorded atomic-claim preview covers all 55 canonical keys through LoD 3', () => {
  const evidence = JSON.parse(readFileSync(new URL('./claims-canonical-browser-snapshots.json', import.meta.url), 'utf8'));
  const canonical = key => {
    const orbit = [];
    for (const ring of [key, [...key].reverse()]) {
      for (let start = 0; start < 4; ++start) orbit.push([...ring.slice(start), ...ring.slice(0, start)].join(','));
    }
    return orbit.sort()[0];
  };
  const expected = new Set();
  for (let a=0;a<=3;++a) for (let b=0;b<=3;++b) for (let c=0;c<=3;++c) for (let d=0;d<=3;++d) {
    expected.add(canonical([a,b,c,d]));
  }
  assert.equal(expected.size, 55);
  assert.equal(evidence.coverage.maximumLod, 3);
  assert.equal(evidence.cases.length, 55);
  const actual = new Set();
  for (const snapshot of evidence.cases) {
    assert.equal(snapshot.passCount, 20);
    assert.equal(snapshot.params.seed, 42);
    assert.equal(new URL(snapshot.manifest).pathname, `/assets/${evidence.manifest}`);
    const requested = ['bottom','right','top','left'].map(k => snapshot.params[k]);
    assert.equal(snapshot.key.join(','), canonical(requested));
    actual.add(snapshot.key.join(','));
    verifyQuadSnapshot(snapshot);
  }
  assert.deepEqual(actual, expected);
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
