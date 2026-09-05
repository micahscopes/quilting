import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {verifyRepairSnapshot} from './repair.verify.mjs';
const read=name=>JSON.parse(readFileSync(new URL(name,import.meta.url),'utf8'));

test('parallel GPU repair preserves every preview mesh and has exact reciprocal adjacency',()=>{
  const current=read('./repair-browser-snapshots.json').cases;
  const baseline=[...read('./claims-canonical-browser-snapshots.json').cases,...read('./claims-browser-snapshots.json').cases];
  assert.equal(current.length,56);
  assert.equal(new Set(current.filter(c=>c.params.seed===42).map(c=>c.key.join())).size,55);
  const triangleSet=c=>c.triangles.map(t=>t.slice().sort((a,b)=>a-b).join(',')).sort();
  let parallelJobs=0;
  for(const c of current){
    verifyRepairSnapshot(c);
    const prev=baseline.find(p=>p.key.join()===c.key.join() && p.params.seed===c.params.seed);
    assert(prev);assert.deepEqual(c.points,prev.points);assert.deepEqual(triangleSet(c),triangleSet(prev));
    if(c.repair[2]>c.repair[6])++parallelJobs;
  }
  assert(parallelJobs>0,'must execute multi-flip rounds, not only a serial degenerate case');
});

test('GPU round-budget exhaustion publishes no draw, and final certification permits complete tiles',()=>{
  const r=read('./repair-budget.browser.json');
  assert.equal(r.passes,34);
  assert.deepEqual(r.observations.map(c=>c.budget),[0,1,64]);
  for(const c of r.observations){
    assert.equal(c.state[4],0);assert.equal(c.state[7],1);assert.equal(c.receipt[6],1);
    if(c.budget<64){assert(c.state[3]>0);assert.equal(c.state[5],0);assert.equal(c.receipt[10],0);assert.equal(c.draw[0],0);}
    else {assert.equal(c.state[3],0);assert.equal(c.state[5],1);assert.equal(c.receipt[10],1);assert.equal(c.draw[0],c.state[1]*3);}
  }
  assert(r.observations[1].state[2]>1,'the first actual GPU repair round contains multiple flips');
});
