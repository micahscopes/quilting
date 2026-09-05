// Offline exact-integer experiment, never imported by the application. Rebuild
// adjacency every round to keep this oracle independent of GPU link mutation.
import assert from 'node:assert/strict';
const orient=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
function incircle(a,b,c,d,crossTerm) {
  const [u,v,w]=[a,b,c].map(p=>{const x=BigInt(p[0]-d[0]),y=BigInt(p[1]-d[1]);return [x,y,x*x+y*y+crossTerm*x*y];});
  return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0]);
}
function precedes(a,b,c,d) {return Math.min(a,b)<Math.min(c,d) || (Math.min(a,b)===Math.min(c,d)&&Math.max(a,b)<Math.max(c,d));}
export function mixedPriority(slot,round) {
  let x=(slot^Math.imul(round,0x9e3779b9))>>>0;
  x=(x^(x>>>16))>>>0;x=Math.imul(x,0x7feb352d)>>>0;
  x=(x^(x>>>15))>>>0;x=Math.imul(x,0x846ca68b)>>>0;
  return (x^(x>>>16))>>>0;
}
export function replayRepairOrder(snapshot,{priority=(slot)=>slot,budget=2048,metric='square'}={}) {
  assert(metric==='square'||metric==='equilateral','unknown plane metric');
  const crossTerm=metric==='equilateral'?1n:0n;
  const {points}=snapshot, triangles=snapshot.triangles.map(t=>t.slice());
  assert(points.length>0 && points.every(p=>p.length===2&&p.every(x=>Number.isInteger(x)&&x>=0&&x<=16384)));
  assert(Number.isInteger(budget)&&budget>0&&budget<=65535);
  let flips=0;const history=[];
  for(let round=0;round<budget;round++) {
    const directed=new Map();
    for(let f=0;f<triangles.length;f++) {
      const t=triangles[f];assert(t.length===3&&t.every(x=>Number.isInteger(x)&&x>=0&&x<points.length));
      assert(orient(...t.map(i=>points[i]))>0);
      for(let e=0;e<3;e++){const key=`${t[e]},${t[(e+1)%3]}`;assert(!directed.has(key));directed.set(key,f*3+e);}
    }
    const twins=triangles.flatMap(t=>t.map((a,e)=>directed.get(`${t[(e+1)%3]},${a}`)??-1));
    const proposals=[];
    for(let slot=0;slot<twins.length;slot++) {
      const twin=twins[slot];if(twin<=slot)continue;
      const f=Math.floor(slot/3),g=Math.floor(twin/3),e=slot%3,h=twin%3;
      const t=triangles[f],s=triangles[g],u=t[e],v=t[(e+1)%3],a=t[(e+2)%3],b=s[(h+2)%3];
      if(orient(points[a],points[u],points[b])<=0||orient(points[b],points[v],points[a])<=0)continue;
      const circle=incircle(points[u],points[v],points[a],points[b],crossTerm);
      if(circle<0n || (circle===0n&&!precedes(a,b,u,v)))continue;
      const footprint=new Set([f,g]);
      for(const face of [f,g])for(let k=0;k<3;k++)if(twins[face*3+k]>=0)footprint.add(Math.floor(twins[face*3+k]/3));
      proposals.push({slot,f,g,a,b,u,v,footprint,rank:priority(slot,round)});
    }
    if(!proposals.length)return {converged:true,rounds:round+1,flips,history,triangles};
    assert.equal(new Set(proposals.map(p=>p.rank)).size,proposals.length,'priority must be injective');
    const claims=new Map();
    for(const p of proposals)for(const f of p.footprint)claims.set(f,Math.min(p.rank,claims.get(f)??Infinity));
    const winners=proposals.filter(p=>[...p.footprint].every(f=>claims.get(f)===p.rank));
    assert(winners.length>0,'strict minimum must win');
    const owned=new Set();
    for(const p of winners){for(const f of p.footprint){assert(!owned.has(f));owned.add(f);}triangles[p.f]=[p.a,p.u,p.b];triangles[p.g]=[p.b,p.v,p.a];}
    flips+=winners.length;history.push({round:round+1,proposals:proposals.length,flips:winners.length});
  }
  return {converged:false,rounds:budget,flips,history,triangles};
}
