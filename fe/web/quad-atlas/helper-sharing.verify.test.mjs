import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

test('loop-helper GPU replay preserves every quad preview case', () => {
  const read = file => JSON.parse(readFileSync(new URL(file, import.meta.url), 'utf8'));
  const evidence = read('./helper-sharing.browser.json');
  const baseline = read('./candidate-index-browser-evidence.json');
  assert.equal(evidence.passes, 47);
  assert.equal(evidence.maximumLod, 3);
  assert.equal(evidence.canonicalKeys, 55);
  assert.equal(evidence.cases.length, 56);
  assert.deepEqual(evidence.cases, baseline.cases);
  assert.equal(evidence.stages.length, 47);
  const delta = evidence.stages.reduce((sum, x) => sum + x.before - x.bytes, 0);
  assert.equal(delta, evidence.reportedWgslBytes.before - evidence.reportedWgslBytes.after);
  assert.deepEqual(evidence.stages.filter(x => x.before !== x.bytes).map(x => x.entry), ['propose', 'retire']);
});
