import {test,expect} from 'bun:test';
import {audit,auditDelaunay,incircleDeterminant} from './packed-audit.mjs';
const Q=16384;
const key={a:0,b:0,c:0,d:0};
function pack(vertices,faces) {
  const ids=faces.flat(),words=new Uint32Array(vertices.length+Math.ceil(ids.length/2));
  vertices.forEach(([x,y],i)=>words[i]=x|(y<<16));
  ids.forEach((v,i)=>words[vertices.length+(i>>1)]|=v<<(16*(i&1)));
  return {status:0,points:vertices.length,triangles:faces.length,words};
}
const tri=()=>pack([[0,0],[Q,0],[0,Q]],[[0,1,2]]);
const quad=()=>pack([[0,0],[Q,0],[Q,Q],[0,Q]],[[0,1,2],[0,2,3]]);
test('exact Delaunay oracle accepts cocircular ties and rejects a geometrically illegal diagonal',()=>{
  expect(auditDelaunay(quad(),true,key)).toEqual({interiorEdges:1,cocircularEdges:1});
  const vertices=[[0,0],[Q/2,0],[Q,0],[Q,Q],[0,Q]];
  const edgeKey={...key,a:1};
  const bad=pack(vertices,[[0,1,3],[1,2,3],[0,3,4]]);
  expect(audit(bad,true,edgeKey)).toBe(2*Q*Q);
  expect(()=>auditDelaunay(bad,true,edgeKey)).toThrow('non-Delaunay edge');
  expect(auditDelaunay(pack(vertices,[[0,1,4],[1,2,3],[1,3,4]]),true,edgeKey).interiorEdges).toBe(2);
});
test('incircle oracle distinguishes square and equilateral metrics exactly',()=>{
  const a=[0,0],b=[Q,0],c=[0,Q],d=[Q,Q];
  expect(incircleDeterminant(a,b,c,d,true)).toBe(0n);
  expect(incircleDeterminant(a,b,c,d,false)).toBe(-(BigInt(Q)**4n));
  expect(incircleDeterminant(b,a,c,d,false)).toBe(BigInt(Q)**4n);
});
test('unit triangle and square satisfy canonical boundaries',()=>{
  expect(audit(tri(),false,key)).toBe(Q*Q);
  expect(audit(quad(),true,key)).toBe(2*Q*Q);
});
test('collinear boundary vertices must be connected, not skipped',()=>{
  const t=pack([[0,0],[Q/2,0],[Q,0],[0,Q]],[[0,1,3],[1,2,3]]);
  expect(audit(t,false,{...key,c:1})).toBe(Q*Q);
  expect(()=>audit(t,false,key)).toThrow('unexpected boundary');
  expect(()=>audit(tri(),false,{...key,c:1})).toThrow('missing boundary point');
});
test('all square symmetries preserve edge keys and orientation when winding is corrected',()=>{
  const p=[[0,0],[Q/2,0],[Q,0],[Q,Q],[0,Q],[Q/2,Q/2]];
  const faces=Array.from({length:5},(_,i)=>[i,(i+1)%5,5]);
  for(let reflect=0;reflect<2;reflect++)for(let rotation=0;rotation<4;rotation++) {
    const transformed=p.map(([x,y])=>{
      if(reflect)x=Q-x;
      for(let r=0;r<rotation;r++)[x,y]=[Q-y,x];
      return [x,y];
    });
    const exponents=[0,0,0,0]; exponents[rotation]=1;
    const t=pack(transformed,faces.map(f=>reflect?[f[0],f[2],f[1]]:f));
    expect(audit(t,true,{a:exponents[0],b:exponents[1],c:exponents[2],d:exponents[3]})).toBe(2*Q*Q);
    expect(auditDelaunay(t,true,{a:exponents[0],b:exponents[1],c:exponents[2],d:exponents[3]}).interiorEdges).toBe(5);
  }
});
test('triangle relabelings permute opposite-edge densities and preserve the boundary',()=>{
  const p=[[0,0],[Q/2,0],[Q,0],[0,Q]],corners=[[0,0],[Q,0],[0,Q]];
  for(const permutation of [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]) {
    const [a,b,c]=permutation.map(i=>corners[i]);
    const transformed=p.map(([x,y])=>[
      a[0]+(b[0]-a[0])*x/Q+(c[0]-a[0])*y/Q,
      a[1]+(b[1]-a[1])*x/Q+(c[1]-a[1])*y/Q,
    ]);
    const sign=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    const faces=[[0,1,3],[1,2,3]].map(f=>sign<0?[f[0],f[2],f[1]]:f);
    const exponents=[0,0,0];exponents[permutation[2]]=1;
    expect(audit(pack(transformed,faces),false,{a:exponents[0],b:exponents[1],c:exponents[2]})).toBe(Q*Q);
    expect(auditDelaunay(pack(transformed,faces),false,{a:exponents[0],b:exponents[1],c:exponents[2]}).interiorEdges).toBe(1);
  }
});
test('corrupted coordinates, winding, padding and multiplicity fail closed',()=>{
  const outside=tri();outside.words[0]=Q|(Q<<16);
  expect(()=>audit(outside,false,key)).toThrow('coordinate');
  const reversed=pack([[0,0],[Q,0],[0,Q]],[[0,2,1]]);
  expect(()=>audit(reversed,false,key)).toThrow('winding');
  const padding=tri();padding.words[4]|=1<<16;
  expect(()=>audit(padding,false,key)).toThrow('padding');
  const repeated=pack([[0,0],[Q,0],[0,Q]],[[0,1,2],[0,1,2]]);
  expect(()=>audit(repeated,false,key)).toThrow('edge incidence');
});
