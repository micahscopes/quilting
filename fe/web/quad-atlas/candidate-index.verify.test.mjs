import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {verifyCandidateTree} from './candidate-index.verify.mjs';

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
