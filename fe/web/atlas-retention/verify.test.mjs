import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {verifyRetention} from './verify.mjs';

test('recorded Fe WebGPU execution passes the independent retention oracle', () => {
  const data = JSON.parse(readFileSync(new URL('./browser-evidence.json', import.meta.url), 'utf8'));
  assert.equal(data.passCount, 6);
  assert.equal(data.deviceStorageBindingLimit, 8);
  assert.equal(verifyRetention(data).retainedAfterScratchReuse, true);
});

// Synthetic inputs test the oracle only; they are not GPU execution evidence.
function expected() {
  return {
    header: [7, 9, 3, 1, 4], copied: [3, 0, 0, 2, 2],
    directory: [
      2,7,100,0,0,3,1,0,
      3,7,0,0,0,0,0,0,
      3,7,0,0,0,0,0,0,
      3,7,103,3,1,3,1,0,
      3,7,104,6,2,3,1,0,
    ],
    resident_points: [10,20,30,11,21,31,12,22,32],
    resident_triangles: [0,1,2], points: [999], triangles: [3],
  };
}
test('retention oracle accepts its independently specified expected state', () => {
  assert.equal(verifyRetention(expected()).retainedAfterScratchReuse, true);
});
for (const [name, mutate] of [
  ['partial publication', d => d.directory[24] = 2],
  ['invalid publication', d => d.directory[32] = 2],
  ['overlapping allocations', d => d.directory[27] = 0],
  ['scratch alias', d => d.resident_points[0] = 999],
  ['stale epoch', d => d.directory[1] = 6],
  ['duplicate completion', d => d.copied[0] = 4],
  ['missing overwrite', d => d.points[0] = 10],
]) {
  test(`retention oracle rejects ${name}`, () => {
    const data = expected(); mutate(data);
    assert.throws(() => verifyRetention(data));
  });
}
