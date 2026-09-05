import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {verifyCandidateTree, verifyCandidateSelection} from './candidate-index.verify.mjs';

test('selection oracle rejects missing neighbor decisions independently of topology', () => {
  const c = {count: 3, candidates: [[0, 0, 0, 25, 0], [3, 0, 0, 1, 1], [5, 0, 0, 1, 2]],
    states: [1, 2, 1]};
  assert.deepEqual(verifyCandidateSelection(c), {accepted: 2});
  assert.throws(() => verifyCandidateSelection({...c, states: [1, 1, 1]}), /accepted conflict/);
  assert.throws(() => verifyCandidateSelection({...c, states: [1, 0, 1]}), /unfinished/);
  assert.deepEqual(verifyCandidateSelection({...c, triangular: true, states: [1, 1, 1]}), {accepted: 3});
});

test('rejected indexed GPU build has correct bounds but fails sample independence', () => {
  const capture = JSON.parse(readFileSync(new URL('./candidate-index-failure.browser.json', import.meta.url), 'utf8'));
  verifyCandidateTree(capture.candidateIndex);
  assert.throws(() => verifyCandidateSelection(capture.candidateIndex), /accepted conflict/);
  assert.equal(capture.receipt[3], 86);
  assert.equal(capture.receipt[10], 0, 'failed repair must not publish a drawable tile');
});

test('fixed indexed GPU build preserves reference selection and every preview mesh', () => {
  const read = file => JSON.parse(readFileSync(new URL(file, import.meta.url), 'utf8'));
  const capture = read('./candidate-index-fixed.browser.json');
  verifyCandidateTree(capture.candidateIndex);
  assert.deepEqual(verifyCandidateSelection(capture.candidateIndex), {accepted: 1});
  assert.equal(capture.receipt[10], 1);
  const evidence = read('./candidate-index-browser-evidence.json');
  const baseline = read('./incidence-mesh-evidence.json');
  assert.equal(evidence.passes, 47);
  assert.equal(evidence.canonicalKeys, 55);
  assert.equal(evidence.maximumLod, 3);
  assert.equal(evidence.cases.length, 56);
  assert.deepEqual(evidence.cases.map(({key, seed, sha256}) => ({key, seed, sha256})), baseline.cases);
});

test('candidate hierarchy model preserves every exact conflict in captured GPU candidate sets', () => {
  // The candidates are actual GPU output; this tree is an independent CPU
  // reference model, not evidence that the new GPU tree has executed yet.
  const baseline = JSON.parse(readFileSync(new URL('./candidate-index-baseline.json', import.meta.url), 'utf8'));
  for (const c of baseline.cases) {
    let leaves = 1; while (leaves * 16 < c.capacity) leaves *= 2;
    const nodes = Array.from({length: 2 * leaves}, () => [0xffffffff, 0xffffffff, 0, 0, 0]);
    for (let s = 0; s < c.count; ++s) {
      const a = nodes[leaves + Math.floor(s / 16)], [x, y, , r] = c.candidates[s];
      a[0] = Math.min(a[0], x); a[1] = Math.min(a[1], y);
      a[2] = Math.max(a[2], x); a[3] = Math.max(a[3], y); a[4] = Math.max(a[4], r);
    }
    for (let i = leaves - 1; i; --i) {
      const a = nodes[i * 2], b = nodes[i * 2 + 1];
      nodes[i] = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]),
        Math.max(a[3], b[3]), Math.max(a[4], b[4])];
    }
    const stats = verifyCandidateTree({...c, words: [0, ...nodes.slice(1).flat()]});
    assert(stats.candidateVisits < stats.exhaustiveCandidateVisits);
    console.log(JSON.stringify({key: c.key, modelOnly: true, ...stats}));
  }
});
