import assert from 'node:assert/strict';
import {verifyQuadSnapshot} from './verify.mjs';

export function verifyRepairSnapshot(snapshot,expectedPassCount=34){
  verifyQuadSnapshot(snapshot);
  const {points,triangles,twins,repair,receipt}=snapshot;
  assert.equal(snapshot.passCount,expectedPassCount);
  assert.equal(twins.length,triangles.length*3);
  assert.deepEqual(repair.slice(0,2),[points.length,triangles.length]);
  assert.deepEqual(repair.slice(2,6),[receipt[12],0,0,1]);
  assert.equal(repair[7],1);
  assert(repair[6]>0 && repair[6]<=64);
  const directed=new Map();
  for(const [face,t] of triangles.entries())for(let e=0;e<3;++e){
    const key=`${t[e]},${t[(e+1)%3]}`;
    assert(!directed.has(key),'directed edges must be unique');
    directed.set(key,face*3+e);
  }
  for(const [face,t] of triangles.entries())for(let e=0;e<3;++e){
    const slot=face*3+e;
    const expected=directed.get(`${t[(e+1)%3]},${t[e]}`)??0xffffffff;
    assert.equal(twins[slot],expected,`wrong neighbor at directed edge ${slot}`);
    if(expected!==0xffffffff)assert.equal(twins[expected],slot);
  }
}
