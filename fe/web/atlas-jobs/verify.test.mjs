import test from 'node:test';
import assert from 'node:assert/strict';
import {expectedQuadCodes,verifyQuadJobCoverage} from './verify.mjs';

// These synthetic fixtures test the oracle only. They are NOT GPU evidence.
function expected(maximum=8) {
  const codes=expectedQuadCodes(maximum);
  return {history:codes.map(c=>c+1),receipt:[codes.length,1,codes.at(-1),0]};
}
test('explicit D4 orbits agree with Burnside counts through LoD 8',()=>{
  for(let maximum=0;maximum<=8;maximum++) {
    const n=maximum+1;
    const count=(n**4+2*n**3+3*n**2+2*n)/8;
    assert.equal(expectedQuadCodes(maximum).length,count);
    assert.equal(verifyQuadJobCoverage(expected(maximum),maximum).canonicalJobs,count);
  }
  assert.equal(expectedQuadCodes().length,1035);
});
test('opposite and adjacent high edges remain distinct square orbits',()=>{
  const encode=edges=>edges.reduce((a,b)=>a*9+b,0);
  const codes=expectedQuadCodes(1);
  assert.ok(codes.includes(encode([0,0,1,1])));
  assert.ok(codes.includes(encode([0,1,0,1])));
});
for(const [name,mutate] of [
  ['missing job',d=>d.history.pop()],
  ['duplicate job',d=>d.history[1]=d.history[0]],
  ['wrong ordinal',d=>[d.history[0],d.history[1]]=[d.history[1],d.history[0]]],
  ['uncleared slot',d=>d.history[700]=0],
  ['premature invalidation',d=>d.receipt[1]=0],
  ['stale valid job after exhaustion',d=>d.receipt[3]=1],
  ['incorrect job count',d=>d.receipt[0]--],
]) test(`coverage oracle rejects ${name}`,()=>{
  const data=expected(); mutate(data);
  assert.throws(()=>verifyQuadJobCoverage(data));
});
test('partial preview cannot masquerade as the full atlas job set',()=>{
  assert.throws(()=>verifyQuadJobCoverage(expected(3)));
});
