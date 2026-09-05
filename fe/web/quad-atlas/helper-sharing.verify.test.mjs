import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compareSamplingQueueTime} from './reference-sampling.probe.mjs';

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

test('helper timing records a paired comparison, not a claimed speedup', () => {
  const evidence = JSON.parse(readFileSync(new URL('./helper-sharing.timing.json', import.meta.url), 'utf8'));
  assert.equal(evidence.runs.length, 2);
  assert.equal(evidence.warmupPairs, 2);
  assert.equal(evidence.measuredPairs, 24);
  for (const run of evidence.runs) {
    assert.equal(run.kind, 'queue-completion-not-gpu-timestamp');
    assert.equal(run.observations.length, 48);
    for (let pair = 0; pair < 24; ++pair) {
      const observations = run.observations.filter(x => x.pair === pair);
      const labels = ['before-helper-sharing', 'after-helper-sharing'];
      assert.deepEqual(observations.map(x => x.implementation), pair % 2 ? labels.reverse() : labels);
      for (const x of observations) {
        assert.ok(Number.isFinite(x.encodingMs) && x.encodingMs >= 0);
        assert.ok(Number.isFinite(x.submitToCompletionMs) && x.submitToCompletionMs >= 0);
      }
    }
  }
});

test('timing rejects ambiguous implementation labels before touching a device', async () => {
  await assert.rejects(compareSamplingQueueTime(null, [], 1, {original: 'same', replacement: 'same'}), /distinct/);
  await assert.rejects(compareSamplingQueueTime(null, [], 0), /positive/);
});
