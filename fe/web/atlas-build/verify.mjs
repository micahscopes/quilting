// Independent readback oracle. Not shipped as an atlas or used for scheduling.
import assert from 'node:assert/strict';
import {expectedQuadCodes} from '../atlas-jobs/verify.mjs';

// Classify reported gates, not geometric truth. Independent geometry replay is
// still required: a GPU's self-reported certificate is not its own oracle.
export function summarizeTileReceipts(data, {shape} = {}) {
  const summary = verifyAtlasDirectory(data, {shape, complete:false});
  const {receipts, directory} = data;
  assert.equal(receipts.length, summary.expected * 16);
  assert.ok(receipts.every(x => Number.isInteger(x) && x >= 0 && x <= 0xffffffff));
  const failures = [];
  for (let i=0; i<summary.expected; i++) {
    const state = directory[i*8];
    if (state !== 2 && state !== 3) continue;
    const r = receipts.slice(i*16, i*16+16);
    const gates = [];
    if (r[0] !== 1 || r[1] !== 1 || r[2] !== 0) gates.push('sampling');
    if (r[6] !== 1 || r[8] !== 0 || r[9] !== 0) gates.push('insertion');
    if (r[14] !== 0) gates.push('repair-invariant');
    if (r[11] !== 1 || r[13] !== 0) gates.push('repair-unsettled');
    if (r[10] !== 1 && !gates.some(g => g.startsWith('repair-'))) gates.push('repair-invalid');
    if (state === 2) {
      assert.deepEqual(gates, [], `ready tile ${i} reports failed gates`);
      assert.equal(r[5], directory[i*8+5], `point receipt at job ${i}`);
      assert.equal(r[7], directory[i*8+6], `triangle receipt at job ${i}`);
    } else {
      failures.push({ordinal:i, gates:gates.length ? gates : ['reservation-or-publication'],
        points:r[5], triangles:r[7], flips:r[12], remainingViolations:r[13]});
    }
  }
  return {...summary, failures};
}

export function verifyAtlasDirectory(data, {shape, complete = true} = {}) {
  assert.ok(shape === 'triangle' || shape === 'quad');
  const keys = shape === 'quad' ? expectedQuadCodes(8) : Array.from({length:165}, (_,i)=>i);
  const {header, directory, completion} = data;
  assert.equal(header.length, 5);
  assert.equal(directory.length, keys.length * 8);
  assert.ok(header.every(Number.isSafeInteger));
  const epoch = header[0];
  assert.ok(epoch > 0);
  let ready = 0, failed = 0, pending = 0, vertices = 0, triangles = 0;
  let lastVertexEnd = 0, lastTriangleEnd = 0;
  for (let i = 0; i < keys.length; i++) {
    const [state, generation, key, vo, to, nv, nt] = directory.slice(i*8, i*8+8);
    assert.equal(generation, epoch, `epoch at job ${i}`);
    assert.ok([0,1,2,3].includes(state), `state at job ${i}`);
    if (state === 0 || state === 1) { pending++; continue; }
    if (state === 3) { failed++; continue; }
    ready++;
    assert.equal(key, keys[i], `canonical identity at job ${i}`);
    assert.ok([vo,to,nv,nt].every(x=>Number.isSafeInteger(x) && x>=0));
    assert.ok(nv >= 3 && nt >= 1);
    assert.ok(vo >= lastVertexEnd && to >= lastTriangleEnd, `overlapping allocation at job ${i}`);
    lastVertexEnd = vo + nv; lastTriangleEnd = to + nt;
    assert.ok(lastVertexEnd <= header[1] && lastTriangleEnd <= header[2]);
    vertices += nv; triangles += nt;
  }
  assert.equal(ready, header[3]); assert.equal(failed, header[4]);
  assert.ok(vertices <= header[1] && triangles <= header[2]);
  if (complete) {
    assert.equal(pending, 0); assert.equal(failed, 0); assert.equal(ready, keys.length);
    assert.equal(vertices, header[1]); assert.equal(triangles, header[2]);
    assert.deepEqual(completion, [epoch, keys.length, 0, 1, 1]);
  }
  return {shape, expected:keys.length, ready, failed, pending, vertices, triangles,
    retainedBytes:(header[1]+header[2])*12, complete:ready===keys.length && failed===0 && pending===0};
}
