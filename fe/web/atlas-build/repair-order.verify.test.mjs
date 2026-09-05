import test from 'node:test';
import assert from 'node:assert/strict';
import {mixedPriority,replayRepairOrder} from './repair-order.verify.mjs';
const square={points:[[0,0],[16384,0],[16384,16384],[0,16384]],triangles:[[0,1,3],[1,2,3]]};
test('exact cocircular tie selects the smaller diagonal and terminates',()=>{
  for(const priority of [slot=>slot,mixedPriority]) {
    const r=replayRepairOrder(square,{priority});
    assert(r.converged);assert.equal(r.flips,1);assert.equal(r.rounds,2);
    assert.deepEqual(r.triangles,[[0,1,2],[2,3,0]]);
  }
});
test('identical coordinate charts can require different Delaunay diagonals',()=>{
  // The equilateral barycentric chart has squared length x*x+y*y+x*y.
  // A Cartesian square is therefore not cocircular under that metric.
  const r=replayRepairOrder(square,{metric:'equilateral'});
  assert(r.converged);assert.equal(r.flips,0);assert.equal(r.rounds,1);
  assert.deepEqual(r.triangles,square.triangles);
  assert.throws(()=>replayRepairOrder(square,{metric:'unspecified'}),/unknown plane metric/);
});
test('mixed priorities are repeatable and injective on a bounded slot corpus',()=>{
  // Not a proof over u32: algebraically each xor-shift and odd multiply is
  // invertible, and xor with a fixed round salt is also a permutation.
  for(const round of [0,1,127,255]) {
    const values=Array.from({length:10000},(_,slot)=>mixedPriority(slot,round));
    assert.equal(new Set(values).size,values.length);
    assert.deepEqual(values,Array.from({length:10000},(_,slot)=>mixedPriority(slot,round)));
  }
});
