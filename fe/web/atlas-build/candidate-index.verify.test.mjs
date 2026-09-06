import test from 'node:test';
import assert from 'node:assert/strict';
import {censusCandidateIndex} from './candidate-index.verify.mjs';
const fixture = () => {
  const w=new Uint32Array(32*8+1);
  for (let s=0;s<32;s++) { w[s*5]=s<16?0:100; w[s*5+3]=1; }
  return w;
};
test('unchanged membership gives identical hierarchy census',()=>{
  const r=censusCandidateIndex(fixture(),32,[0,16]);
  for (const q of r.queries) {assert.deepEqual(q.all,q.initiallyAdmissible);assert.equal(q.all.returned,16);}
});
test('rejected large-radius candidates can contaminate an otherwise distant subtree',()=>{
  const w=fixture(); w[31*5+3]=20000; w[32*5+31]=2; w[32*6+31]=2;
  const q=censusCandidateIndex(w,32,[0]).queries[0];
  assert.equal(q.all.returned,32); assert.equal(q.initiallyAdmissible.returned,16);
});
test('live large-radius candidates must remain represented',()=>{
  const w=fixture();w[31*5+3]=20000;
  const q=censusCandidateIndex(w,32,[0]).queries[0];
  assert.deepEqual(q.all,q.initiallyAdmissible);assert.equal(q.all.returned,32);
});
test('reject non-initial state and unbounded query requests',()=>{
  const w=fixture();w[32*8]=1;
  assert.throws(()=>censusCandidateIndex(w,32,[0]),/initialized/);
  assert.throws(()=>censusCandidateIndex(fixture(),32,[32]),/query slots/);
  assert.throws(()=>censusCandidateIndex(fixture(),32,Array(257).fill(0)),/query slots/);
  const accepted=fixture();accepted[32*5]=1;
  assert.throws(()=>censusCandidateIndex(accepted,32,[0]),/initial immutable/);
});
