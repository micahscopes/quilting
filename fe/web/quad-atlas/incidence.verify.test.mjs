import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {verifyRepairSnapshot} from './repair.verify.mjs';

// Independent finite placement model. Actual shader execution is checked
// separately, including the index captured *before* repair changes the faces.
function build(vertices,triangles,tile,reverse){
  const order=n=>Array.from({length:n},(_,i)=>reverse?n-1-i:i);
  const degree=Array(vertices).fill(0);
  for(const f of order(triangles.length))for(const v of triangles[f])++degree[v];
  const offsets=[],totals=[],base=[];
  const blocks=Math.ceil(vertices/tile);
  for(const b of order(blocks)){
    let sum=0;for(let v=b*tile;v<Math.min(vertices,(b+1)*tile);++v){offsets[v]=sum;sum+=degree[v];}totals[b]=sum;
  }
  let total=0;for(let b=0;b<blocks;++b){base[b]=total;total+=totals[b];}
  for(const v of order(vertices))offsets[v]+=base[Math.floor(v/tile)];offsets[vertices]=total;
  const cursor=Array(vertices).fill(0),edges=[];
  for(const slot of order(triangles.length*3)){
    const v=triangles[Math.floor(slot/3)][slot%3],dest=offsets[v]+cursor[v]++;
    assert(dest<offsets[v+1]);assert.equal(edges[dest],undefined);edges[dest]=slot;
  }
  return {offsets,edges};
}

export function verifyIncidence({points,triangles,index}){
  assert.equal(index.offsets.length,points.length+1);
  assert.equal(index.offsets[0],0);
  assert.equal(index.offsets.at(-1),triangles.length*3);
  assert.equal(index.edges.length,triangles.length*3);
  const used=new Set();
  for(let v=0;v<points.length;++v){
    const begin=index.offsets[v],end=index.offsets[v+1];
    assert(begin<=end);
    const expected=triangles.flatMap((t,f)=>t.flatMap((u,e)=>u===v?[f*3+e]:[]));
    const observed=index.edges.slice(begin,end);
    assert.deepEqual(observed.slice().sort((a,b)=>a-b),expected);
    for(const slot of observed){assert(!used.has(slot));used.add(slot);}
  }
  assert.equal(used.size,triangles.length*3);
}

test('actual GPU incidence covers every original directed edge exactly once',()=>{
  const r=JSON.parse(readFileSync(new URL('./incidence-browser.json',import.meta.url),'utf8'));
  assert.equal(r.passes,44);
  const initial=r.observations.find(o=>o.budget===0);
  assert.equal(initial.state[4],0);assert.equal(initial.draw[0],0);
  verifyIncidence(initial.geometry);
  assert(initial.state[3]>0,'incidence validity must not claim Delaunay convergence');
  assert.equal(r.observations.at(-1).state[5],1);
});

test('parallel incidence leaves every preview mesh, neighbor link and repair receipt unchanged',()=>{
  const read=name=>JSON.parse(readFileSync(new URL(name,import.meta.url),'utf8'));
  const evidence=read('./incidence-mesh-evidence.json'),baseline=read('./repair-browser-snapshots.json').cases;
  assert.equal(evidence.passes,44);assert.equal(evidence.maximumLod,3);assert.equal(evidence.canonicalKeys,55);
  assert.equal(evidence.cases.length,56);
  assert.deepEqual(evidence.cases.map(c=>[c.key,c.seed]),baseline.map(c=>[c.key,c.params.seed]));
  for(const c of evidence.cases){
    const p=baseline.find(p=>p.key.join()===c.key.join() && p.params.seed===c.seed);
    verifyRepairSnapshot(p);
    const payload={points:p.points,triangles:p.triangles,twins:p.twins,repair:p.repair,receipt:p.receipt};
    assert.equal(c.sha256,createHash('sha256').update(JSON.stringify(payload)).digest('hex'));
  }
});

test('parallel GPU face validation rejects degenerate, out-of-range, reversed and invalid input',()=>{
  const cases=JSON.parse(readFileSync(new URL('./incidence-faults.browser.json',import.meta.url),'utf8'));
  assert.equal(cases.length,4);
  for(const c of cases){
    assert.equal(c.passes,44);
    const o=c.observations[0];assert.equal(o.budget,64);
    assert.equal(o.receipt[6],1,'insertion succeeded before fault injection');
    assert(o.state[4]>0);assert.equal(o.state[5],0);assert.equal(o.receipt[10],0);assert.equal(o.draw[0],0);
  }
});

test('vertex ranges and edge placement are independent of parallel arrival order',()=>{
  const corpus=JSON.parse(readFileSync(new URL('./repair-browser-snapshots.json',import.meta.url),'utf8')).cases;
  const wheel=n=>({points:Array(n+1),triangles:Array.from({length:n},(_,i)=>[0,i+1,(i+1)%n+1])});
  for(const mesh of [...corpus,wheel(3),wheel(64),wheel(257)]){
    let reference;
    for(const tile of [1,2,3,16,64,512])for(const reverse of [false,true]){
      const index=build(mesh.points.length,mesh.triangles,tile,reverse);
      verifyIncidence({...mesh,index});
      if(reference)assert.deepEqual(index.offsets,reference.offsets);else reference=index;
    }
  }
});
