// Independent diagnostic oracle, never loaded by the production demo.
import assert from 'node:assert/strict';

const LEAF = 16;
const empty = () => [0xffffffff, 0xffffffff, 0, 0, 0];
const union = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]),
  Math.max(a[2], b[2]), Math.max(a[3], b[3]), Math.max(a[4], b[4])];
const after = node => {while (node > 1 && node % 2) node = Math.floor(node / 2); return node <= 1 ? 0 : node + 1;};

export function verifyCandidateTree({capacity, count, words, candidates, triangular = false}) {
  let leaves = 1; while (leaves * LEAF < capacity) leaves *= 2;
  assert.equal(words.length, 1 + (2 * leaves - 1) * 5);
  assert.equal(words[0], 0, 'every parent level must have completed');
  assert.equal(candidates.length, count);
  const node = id => words.slice(1 + (id - 1) * 5, 1 + id * 5);
  for (let leaf = 0; leaf < leaves; ++leaf) {
    let expected = empty();
    for (let slot = leaf * LEAF; slot < Math.min(count, (leaf + 1) * LEAF); ++slot) {
      const [x, y, , radius] = candidates[slot];
      assert(x >= 0 && x <= 16384 && y >= 0 && y <= 16384);
      expected = union(expected, [x, y, x, y, radius]);
    }
    assert.deepEqual(node(leaves + leaf), expected, `leaf ${leaf}`);
  }
  for (let id = leaves - 1; id; --id)
    assert.deepEqual(node(id), union(node(id * 2), node(id * 2 + 1)), `parent ${id}`);

  let visits = 0, slots = 0, conflicts = 0;
  for (const [x, y, , radius] of candidates) {
    // Reference walk examines whole leaves; the Fe cursor yields individual
    // slots between calls. Verify coverage against every exact pair, not just
    // accepted output points (which could conceal missing search neighbors).
    const observed = new Set(); let id = 1;
    while (id) {
      ++visits;
      const [xmin, ymin, xmax, ymax, r] = node(id);
      const dx = Math.max(xmin - x, x - xmax, 0), dy = Math.max(ymin - y, y - ymax, 0);
      if (xmin === 0xffffffff || dx * dx + dy * dy >= Math.max(radius, r)) id = after(id);
      else if (id < leaves) id *= 2;
      else {
        for (let s = (id - leaves) * LEAF; s < Math.min(count, (id - leaves + 1) * LEAF); ++s) {
          assert(!observed.has(s)); observed.add(s); ++slots;
        }
        id = after(id);
      }
    }
    for (let s = 0; s < count; ++s) {
      const [a, b, , otherRadius] = candidates[s], dx = a - x, dy = b - y;
      const d2 = triangular ? 3 * (dx * dx + dy * dy + dx * dy) : dx * dx + dy * dy;
      if (d2 < Math.max(radius, otherRadius)) {
        ++conflicts; assert(observed.has(s), `missed exact conflict ${s}`);
      }
    }
  }
  return {candidates: count, leaves, nodeVisits: visits, candidateVisits: slots,
    exhaustiveCandidateVisits: count * count, exactConflicts: conflicts};
}
