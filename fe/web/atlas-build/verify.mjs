// Independent readback oracle. Not shipped as an atlas or used for scheduling.
import assert from 'node:assert/strict';
import {expectedQuadCodes} from '../atlas-jobs/verify.mjs';

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
