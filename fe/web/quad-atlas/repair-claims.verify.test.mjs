import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

function select(footprints,order){
  const claims=new Map();
  for(const id of order)for(const face of footprints[id])claims.set(face,Math.min(id,claims.get(face)??Infinity));
  return footprints.flatMap((faces,id)=>faces.every(f=>claims.get(f)===id)?[id]:[]);
}
function disjoint(a,b){return a.every(x=>!b.includes(x));}

test('closed-neighborhood claims have disjoint winners and make progress in every small hypergraph',()=>{
  // All nonempty subsets of four faces, three proposals, all six arrivals.
  const sets=Array.from({length:15},(_,i)=>[0,1,2,3].filter(f=>(i+1)&(1<<f)));
  const orders=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
  for(const a of sets)for(const b of sets)for(const c of sets){
    const footprints=[a,b,c],reference=select(footprints,orders[0]);
    assert(reference.includes(0),'minimum candidate owns every requested face');
    for(const order of orders)assert.deepEqual(select(footprints,order),reference);
    for(let i=0;i<reference.length;++i)for(let j=i+1;j<reference.length;++j)
      assert(disjoint(footprints[reference[i]],footprints[reference[j]]));
  }
});

function index(triangles){
  const directed=new Map(),twins=Array(triangles.length*3).fill(-1);
  for(const [face,t] of triangles.entries())for(let e=0;e<3;++e)directed.set(`${t[e]},${t[(e+1)%3]}`,face*3+e);
  for(const [face,t] of triangles.entries())for(let e=0;e<3;++e)twins[face*3+e]=directed.get(`${t[(e+1)%3]},${t[e]}`)??-1;
  return twins;
}
function footprint(twins,slot){
  const faces=[Math.floor(slot/3),Math.floor(twins[slot]/3)],closed=[...faces];
  for(const f of faces)for(let e=0;e<3;++e)if(twins[f*3+e]>=0)closed.push(Math.floor(twins[f*3+e]/3));
  return {faces,closed:[...new Set(closed)]};
}

test('captured mesh neighborhoods expose why face-disjoint flip selection is insufficient',()=>{
  const cases=JSON.parse(readFileSync(new URL('./claims-canonical-browser-snapshots.json',import.meta.url),'utf8')).cases;
  let unsafeFaceOnlyPairs=0,multiWinnerMeshes=0;
  for(const c of cases){
    const twins=index(c.triangles);
    const proposals=twins.flatMap((t,s)=>t>s?[footprint(twins,s)]:[]);
    for(let i=0;i<proposals.length;++i)for(let j=i+1;j<proposals.length;++j){
      if(disjoint(proposals[i].faces,proposals[j].faces) && !disjoint(proposals[i].closed,proposals[j].closed))
        ++unsafeFaceOnlyPairs;
    }
    const selected=select(proposals.map(p=>p.closed),proposals.map((_,i)=>i));
    if(selected.length>1)++multiWinnerMeshes;
    for(let i=0;i<selected.length;++i)for(let j=i+1;j<selected.length;++j)
      assert(disjoint(proposals[selected[i]].closed,proposals[selected[j]].closed));
  }
  assert(unsafeFaceOnlyPairs>0);
  assert(multiWinnerMeshes>0,'conservative ownership must still allow parallel work');
});
