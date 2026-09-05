import test from 'node:test';
import assert from 'node:assert/strict';
import {verifyAtlasDirectory, summarizeTileReceipts} from './verify.mjs';
import {expectedQuadCodes} from '../atlas-jobs/verify.mjs';

// Synthetic records test the oracle, not GPU geometry or runtime performance.
function records(shape) {
  const keys=shape==='quad'?expectedQuadCodes(8):Array.from({length:165},(_,i)=>i);
  return {header:[1,keys.length*4,keys.length*2,keys.length,0],
    directory:keys.flatMap((key,i)=>[2,1,key,i*4,i*2,4,2,0]),
    completion:[1,keys.length,0,1,1]};
}
function receiptRecords() {
  return {...records('triangle'), receipts:Array.from({length:165},()=>
    [1,1,0,0,4,4,1,2,0,0,1,1,0,0,0,0]).flat()};
}
test('classify repair exhaustion separately from sampling and insertion',()=>{
  const d=receiptRecords(); d.directory[0]=3; d.header[3]--; d.header[4]++;
  d.receipts[10]=0; d.receipts[11]=0; d.receipts[13]=17;
  assert.deepEqual(summarizeTileReceipts(d,{shape:'triangle'}).failures[0].gates,['repair-unsettled']);
});
test('do not trust a ready record with a failed receipt',()=>{
  const d=receiptRecords(); d.receipts[14]=1;
  assert.throws(()=>summarizeTileReceipts(d,{shape:'triangle'}),/ready tile/);
});
test('reject mismatched receipt geometry counts',()=>{
  const d=receiptRecords(); d.receipts[5]=5;
  assert.throws(()=>summarizeTileReceipts(d,{shape:'triangle'}),/point receipt/);
});
test('retain unknown publication failures instead of calling them repair failures',()=>{
  const d=receiptRecords(); d.directory[0]=3; d.header[3]--; d.header[4]++;
  assert.deepEqual(summarizeTileReceipts(d,{shape:'triangle'}).failures[0].gates,['reservation-or-publication']);
});
for (const shape of ['triangle','quad']) {
  test(`${shape}: complete coverage requires all canonical keys, not a ladder`,()=>{
    assert.equal(verifyAtlasDirectory(records(shape),{shape}).complete,true);
  });
  for (const [name,mutate] of [
    ['missing tile',d=>{d.directory[0]=0;d.header[3]--;}],
    ['failed tile',d=>{d.directory[0]=3;d.header[3]--;d.header[4]++;}],
    ['incorrect key',d=>d.directory[10]=d.directory[2]],
    ['stale epoch',d=>d.directory[1]=2],
    ['overlapping vertices',d=>d.directory[11]=0],
    ['overlapping triangles',d=>d.directory[12]=0],
    ['forged completion',d=>d.completion[4]=0],
    ['forged ready count',d=>d.header[3]--],
  ]) test(`${shape}: reject ${name}`,()=>{
    const data=records(shape);mutate(data);
    assert.throws(()=>verifyAtlasDirectory(data,{shape}));
  });
}
