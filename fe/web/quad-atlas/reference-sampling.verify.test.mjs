import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';

test('manual sampling probe preserves all preview meshes without replacing the remaining Fe pipeline', () => {
  const read = file => JSON.parse(readFileSync(new URL(file, import.meta.url), 'utf8'));
  const evidence = read('./reference-sampling.browser.json');
  const source = readFileSync(new URL('./reference-sampling.wgsl', import.meta.url));
  assert.equal(source.length, evidence.manualBytes);
  assert.equal(createHash('sha256').update(source).digest('hex'), evidence.sourceSha256);
  assert.equal(evidence.kind, 'manual-proposal-retirement-only');
  assert.deepEqual(evidence.generatedStages, ['propose', 'retire']);
  assert.equal(evidence.cases.length, 56);
  const baseline = read('./candidate-index-browser-evidence.json');
  assert.deepEqual(evidence.cases, baseline.cases);
  assert(!readFileSync(new URL('./index.html', import.meta.url), 'utf8').includes('reference-sampling'));
  for (const timing of evidence.timings) {
    assert.equal(timing.kind, 'queue-completion-not-gpu-timestamp');
    assert.equal(timing.observations.length, 24);
    for (let pair = 0; pair < 12; ++pair) {
      const rows = timing.observations.filter(o => o.pair === pair);
      assert.deepEqual(rows.map(o => o.implementation).sort(), ['fe', 'manual']);
      assert(rows.every(o => o.submitToCompletionMs >= 0 && o.encodingMs >= 0));
    }
  }
});
