// Independent readback oracle, not part of the demo or its render pipeline.
// Browser QA supplies canonical Q14 points and resident triangles.
import assert from 'node:assert/strict';

const S = 16384;
const orient = (a, b, c) =>
  (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
const edgeKey = (a, b) => a < b ? `${a}:${b}` : `${b}:${a}`;
const between = (a, b, p) => orient(a, b, p) === 0 &&
  p[0] >= Math.min(a[0], b[0]) && p[0] <= Math.max(a[0], b[0]) &&
  p[1] >= Math.min(a[1], b[1]) && p[1] <= Math.max(a[1], b[1]);
function intersect(a, b, c, d) {
  const x = orient(a, b, c), y = orient(a, b, d);
  const z = orient(c, d, a), w = orient(c, d, b);
  return (x * y < 0 && z * w < 0) ||
    between(a, b, c) || between(a, b, d) || between(c, d, a) || between(c, d, b);
}
function incircle(a, b, c, d, crossTerm) {
  const rows = [a, b, c].map(p => {
    const x = BigInt(p[0] - d[0]), y = BigInt(p[1] - d[1]);
    return [x, y, x * x + y * y + BigInt(crossTerm) * x * y];
  });
  const [u, v, w] = rows;
  return u[0] * (v[1] * w[2] - v[2] * w[1]) -
    u[1] * (v[0] * w[2] - v[2] * w[0]) +
    u[2] * (v[0] * w[1] - v[1] * w[0]);
}

export function verifyQuadSnapshot({ key, points, triangles, receipt }) {
  assert.equal(key.length, 4);
  assert(key.every(k => Number.isInteger(k) && k >= 0 && k <= 8));
  const boundary = key.reduce((sum, k) => sum + 2 ** k, 0);
  if (receipt) {
    for (const lane of [0, 1, 6, 10, 11]) assert.equal(receipt[lane], 1, `success lane ${lane}`);
    for (const lane of [2, 8, 9, 13, 14]) assert.equal(receipt[lane], 0, `failure lane ${lane}`);
    assert.equal(receipt[4], boundary);
    assert.equal(receipt[5], points.length);
    assert.equal(receipt[7], triangles.length);
  }
  assert.equal(new Set(points.map(p => p.join(','))).size, points.length, 'duplicate points');
  assert(points.every(p => p.length === 2 && p.every(x => Number.isInteger(x) && x >= 0 && x <= S)));
  let id = 0;
  for (let e = 0; e < 4; ++e) {
    const n = 2 ** key[e];
    for (let i = 0; i < n; ++i) {
      const t = S * i / n;
      const expected = [[t, 0], [S, t], [S - t, S], [0, S - t]][e];
      assert.deepEqual(points[id++], expected, `boundary point ${id - 1}`);
    }
  }
  return verifyPlanarMesh({points,triangles,boundaryCycle:Array.from({length:boundary},(_,i)=>i),area2:2*S*S,crossTerm:0});
}

// Independent exact topology/geometry checks shared by the square and
// equilateral-triangle reference domains. crossTerm selects x²+y² or x²+xy+y².
export function verifyPlanarMesh({points,triangles,boundaryCycle,area2:expectedArea2,crossTerm}) {
  assert(crossTerm===0 || crossTerm===1);
  const boundary=boundaryCycle.length;
  assert.equal(new Set(points.map(p=>p.join(','))).size,points.length,'duplicate points');
  assert(points.every(p=>p.length===2 && p.every(x=>Number.isInteger(x) && x>=0 && x<=S)));
  assert.equal(triangles.length, 2 * points.length - boundary - 2, 'Euler disk relation');
  const edges = new Map(), used = new Set();
  let area2 = 0;
  for (const t of triangles) {
    assert.equal(t.length, 3);
    assert(t.every(i => Number.isInteger(i) && i >= 0 && i < points.length));
    const area = orient(...t.map(i => points[i]));
    assert(area > 0, 'degenerate or reversed triangle');
    area2 += area;
    for (let e = 0; e < 3; ++e) {
      const a = t[e], b = t[(e + 1) % 3], opposite = t[(e + 2) % 3];
      used.add(a);
      const k = edgeKey(a, b);
      if (!edges.has(k)) edges.set(k, []);
      edges.get(k).push([a, b, opposite]);
    }
  }
  assert.equal(used.size, points.length, 'unreferenced vertex');
  assert.equal(area2, expectedArea2, 'exact domain area');
  const hull = new Set(boundaryCycle.map((id,i)=>edgeKey(id,boundaryCycle[(i+1)%boundary])));
  for (const k of hull) assert(edges.has(k), `missing boundary segment ${k}`);
  for (const [k, owners] of edges) {
    assert.equal(owners.length, hull.has(k) ? 1 : 2, `edge incidence ${k}`);
    if (owners.length === 2) {
      const [a, b, c] = owners[0], [u, v, d] = owners[1];
      assert(a === v && b === u, `edge orientation ${k}`);
      // Only convex quadrilaterals admit an alternate diagonal.
      if (orient(points[c], points[d], points[a]) * orient(points[c], points[d], points[b]) < 0) {
        assert(incircle(points[a], points[b], points[c], points[d],crossTerm) <= 0n, `Delaunay violation ${k}`);
      }
    }
  }
  const unique = [...edges.values()].map(owners => owners[0]);
  for (let i = 0; i < unique.length; ++i) {
    for (let j = i + 1; j < unique.length; ++j) {
      const [a, b] = unique[i], [c, d] = unique[j];
      if (a === c || a === d || b === c || b === d) continue;
      assert(!intersect(points[a], points[b], points[c], points[d]), `crossing edges ${i}, ${j}`);
    }
  }
  return { points: points.length, triangles: triangles.length, boundary, edges: edges.size };
}
