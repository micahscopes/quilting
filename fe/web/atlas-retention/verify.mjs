import assert from 'node:assert/strict';

// Independent expected results for the Fe-authored retention fixture.
export function verifyRetention(data) {
  assert.deepEqual(data.header, [7, 9, 3, 1, 4]);
  assert.deepEqual(data.copied, [3, 0, 0, 2, 2]);
  assert.equal(data.directory.length, 40);
  const records = Array.from({length: 5}, (_, i) => data.directory.slice(i * 8, i * 8 + 8));
  assert.deepEqual(records.map(r => r[0]), [2, 3, 3, 3, 3]);
  assert.ok(records.every(r => r[1] === 7));
  assert.deepEqual(records[0], [2, 7, 100, 0, 0, 3, 1, 0]);
  assert.deepEqual(records[3].slice(2, 7), [103, 3, 1, 3, 1]);
  assert.deepEqual(records[4].slice(2, 7), [104, 6, 2, 3, 1]);
  assert.deepEqual(data.resident_points.slice(0, 9), [10, 20, 30, 11, 21, 31, 12, 22, 32]);
  assert.deepEqual(data.resident_triangles.slice(0, 3), [0, 1, 2]);
  assert.equal(data.points[0], 999, 'scratch was actually overwritten');
  assert.equal(data.triangles[0], 3, 'invalid local index was actually injected');
  return {ready: 1, failed: 4, retainedAfterScratchReuse: true};
}
