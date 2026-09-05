import test from 'node:test';
import assert from 'node:assert/strict';

// Independent finite model of PIPELINE.md's insertion ownership contract.
// This is not GPU execution evidence or a production topology implementation.
const none = 0xffffffff;
const faceCount = 4;

function precedingConflict(proposals) {
  return proposals.map((faces, point) => faces.length > 0 &&
    !proposals.slice(0, point).some(other => faces.some(face => other.includes(face))));
}

function claimed(proposals, arrival, rank = point => point, decode = word => word) {
  const claims = Array(faceCount).fill(none);
  for (const point of arrival) {
    for (const face of proposals[point]) claims[face] = Math.min(claims[face], rank(point));
  }
  const winners = proposals.map((faces, point) => faces.length > 0 &&
    faces.every(face => claims[face] === rank(point)));
  const owners = claims.map(decode).map(point => point !== none && winners[point] ? point : none);
  return { winners, owners };
}

function* permutations(values) {
  if (!values.length) { yield []; return; }
  for (const value of values) {
    for (const suffix of permutations(values.filter(other => other !== value))) {
      yield [value, ...suffix];
    }
  }
}

test('per-face minimum claims preserve preceding-conflict choices for every small proposal set', () => {
  // Empty represents any ineligible point. A singleton covers a strict face
  // or boundary edge; a pair covers the two sides of an interior edge.
  const choices = [[]];
  for (let a = 0; a < faceCount; ++a) {
    choices.push([a]);
    for (let b = a + 1; b < faceCount; ++b) choices.push([a, b]);
  }
  const orders = [...permutations([0, 1, 2, 3])];
  for (let encoded = 0; encoded < choices.length ** 4; ++encoded) {
    let cursor = encoded;
    const proposals = Array.from({ length: 4 }, () => {
      const faces = choices[cursor % choices.length];
      cursor = Math.floor(cursor / choices.length);
      return faces;
    });
    const expected = precedingConflict(proposals);
    const expectedOwners = Array.from({ length: faceCount }, (_, face) => {
      const owners = proposals.flatMap((faces, point) => expected[point] && faces.includes(face) ? [point] : []);
      assert.ok(owners.length <= 1, 'winning insertions must be face-disjoint');
      return owners[0] ?? none;
    });
    assert.equal(expected.some(Boolean), proposals.some(faces => faces.length > 0),
      'every nonempty valid proposal set makes progress');
    for (const order of orders) {
      const actual = claimed(proposals, order);
      assert.deepEqual(actual.winners, expected);
      assert.deepEqual(actual.owners, expectedOwners);
    }
  }
});

// Deliberately independent of the Fe implementation's mask/shift sequence.
const reverseBits = value => Number.parseInt(value.toString(2).padStart(32, '0').split('').reverse().join(''), 2);

test('dyadic ranks preserve point identity and use the same empty-claim sentinel', () => {
  assert.equal(reverseBits(none), none);
  for (let point = 0; point < 65536; ++point) {
    assert.equal(reverseBits(reverseBits(point)), point);
  }
  for (const point of [0x80000000, 0xdeadbeef, 0xfffffffe]) {
    assert.equal(reverseBits(reverseBits(point)), point);
  }
});

test('ranked claims decode stable owners and reject half-won edges', () => {
  const ranks = [0, 1, 2, 3].map(reverseBits);
  const choices = [[], [0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
  const orders = [...permutations([0, 1, 2, 3])];
  const decode = new Map(ranks.map((rank, point) => [rank, point]));
  decode.set(none, none);
  for (let encoded = 0; encoded < choices.length ** 4; ++encoded) {
    let cursor = encoded;
    const proposals = Array.from({ length: 4 }, () => {
      const faces = choices[cursor % choices.length];
      cursor = Math.floor(cursor / choices.length);
      return faces;
    });
    const expected = proposals.map((faces, point) => faces.length > 0 &&
      !proposals.some((other, id) => ranks[id] < ranks[point] && faces.some(face => other.includes(face))));
    const owners = Array.from({ length: faceCount }, (_, face) => {
      const point = proposals.findIndex((faces, id) => expected[id] && faces.includes(face));
      return point < 0 ? none : point;
    });
    for (const order of orders) {
      const actual = claimed(proposals, order, point => ranks[point], rank => decode.get(rank));
      assert.deepEqual(actual.winners, expected);
      assert.deepEqual(actual.owners, owners);
    }
  }
});

test('ordered boundary insertion exceeds the round cap; dyadic priority balances isolated edges', () => {
  const ranks = Array.from({ length: 2049 }, (_, point) => reverseBits(point));
  function rounds(first, last, priority) {
    if (last - first <= 1) return 0;
    let best = first + 1;
    for (let point = first + 2; point < last; ++point) {
      if (priority(point) < priority(best)) best = point;
    }
    return 1 + Math.max(rounds(first, best, priority), rounds(best, last, priority));
  }
  for (let lod = 0; lod <= 8; ++lod) {
    const segments = 2 ** lod;
    assert.equal(rounds(0, segments, point => point), segments - 1);
    assert.equal(rounds(0, segments, point => ranks[point]), lod);
    // Mixed edge LoDs make boundary-prefix offsets unaligned. Test those too;
    // this remains an isolated-edge model, not a whole-mesh round bound.
    for (let offset = 0; offset <= 1024; ++offset) {
      assert.ok(rounds(offset, offset + segments, point => ranks[point]) <= lod + 1);
    }
  }
});

test('a half-won interior edge is not published, nor treated as a maximal independent set', () => {
  const proposals = [[0], [0, 1], [1, 2]];
  assert.deepEqual(claimed(proposals, [2, 1, 0]), {
    winners: [true, false, false], owners: [0, none, none, none],
  });
});
