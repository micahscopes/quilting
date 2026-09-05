import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {verifyPlanarMesh} from '../quad-atlas/verify.mjs';
import {verifyCandidateTree, verifyCandidateSelection} from '../quad-atlas/candidate-index.verify.mjs';

for(const [filename,passes] of [['repair-browser-snapshot.json',34],['incidence-browser-snapshot.json',44],['candidate-index-browser-snapshot.json',47]]){
test(`GPU triangle (${passes} passes) respects its barycentric boundary and equilateral Delaunay metric`,()=>{
  const c=JSON.parse(readFileSync(new URL(filename,import.meta.url),'utf8'));
  assert.equal(c.passCount,passes);
  if(passes===47){
    verifyCandidateTree(c.candidateIndex);
    assert.deepEqual(verifyCandidateSelection(c.candidateIndex),{accepted:5});
    const baseline=JSON.parse(readFileSync(new URL('incidence-browser-snapshot.json',import.meta.url),'utf8'));
    for(const field of ['points','triangles','twins','repair','receipt'])assert.deepEqual(c[field],baseline[field]);
  }
  const S=16384,[a,b,d]=c.key.map(x=>2**x),boundary=[];
  for(let i=0;i<=d;++i)boundary.push([S-S*i/d,S*i/d,0]);
  for(let i=1;i<=b;++i)boundary.push([S-S*i/b,0,S*i/b]);
  for(let i=1;i<a;++i)boundary.push([0,S-S*i/a,S*i/a]);
  assert.deepEqual(c.points.slice(0,boundary.length),boundary);
  assert(c.points.every(p=>p.length===3 && p.every(x=>Number.isInteger(x)&&x>=0) && p.reduce((a,b)=>a+b,0)===S));
  const boundaryCycle=[...Array.from({length:d+1},(_,i)=>i),
    ...Array.from({length:a-1},(_,i)=>d+b+1+i),
    ...Array.from({length:b},(_,i)=>d+b-i)];
  const geometry=verifyPlanarMesh({points:c.points.map(p=>p.slice(1)),triangles:c.triangles,boundaryCycle,area2:S*S,crossTerm:1});
  assert.deepEqual(geometry,{points:19,triangles:22,boundary:14,edges:40});
  assert.deepEqual(c.repair.slice(0,2),[19,22]);
  assert.deepEqual(c.repair.slice(3,6),[0,0,1]);assert.equal(c.repair[7],1);
  for(const lane of [0,1,7,15,16,17])assert.equal(c.receipt[lane],1);
  assert.deepEqual(c.receipt.slice(18),[19,22,c.repair[2],0,0]);
  const edges=new Map();
  for(const [face,t] of c.triangles.entries())for(let e=0;e<3;++e)edges.set(`${t[e]},${t[(e+1)%3]}`,face*3+e);
  assert.equal(c.twins.length,c.triangles.length*3);
  for(const [face,t] of c.triangles.entries())for(let e=0;e<3;++e)
    assert.equal(c.twins[face*3+e],edges.get(`${t[(e+1)%3]},${t[e]}`)??0xffffffff);
});
}
