import test from 'node:test';
import assert from 'node:assert/strict';
import {verifyAtlasDirectory} from './verify.mjs';
import {expectedQuadCodes} from '../atlas-jobs/verify.mjs';

// Synthetic records test the oracle, not GPU geometry or runtime performance.
function records(shape) {
  const keys=shape==='quad'?expectedQuadCodes(8):Array.from({length:165},(_,i)=>i);
  return {header:[1,keys.length*4,keys.length*2,keys.length,0],
    directory:keys.flatMap((key,i)=>[2,1,key,i*4,i*2,4,2,0]),
    completion:[1,keys.length,0,1,1]};
}
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
