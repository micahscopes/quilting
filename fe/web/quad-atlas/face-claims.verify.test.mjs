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

function claimed(proposals, arrival) {
  const claims = Array(faceCount).fill(none);
  for (const point of arrival) {
    for (const face of proposals[point]) claims[face] = Math.min(claims[face], point);
  }
  const winners = proposals.map((faces, point) => faces.length > 0 &&
    faces.every(face => claims[face] === point));
  const owners = claims.map(point => point !== none && winners[point] ? point : none);
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

test('a half-won interior edge is not published, nor treated as a maximal independent set', () => {
  const proposals = [[0], [0, 1], [1, 2]];
  assert.deepEqual(claimed(proposals, [2, 1, 0]), {
    winners: [true, false, false], owners: [0, none, none, none],
  });
});
