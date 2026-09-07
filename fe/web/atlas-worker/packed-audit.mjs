// Independent test oracle, not atlas generation or application policy.
// Check the packed mesh and the exact dyadic edge contract, not just its area.
const Q=16384;
const edgeId=(a,b)=>Math.min(a,b)*65536+Math.max(a,b);
const pointWord=(x,y)=>(x|(y<<16))>>>0;

// Exact independent oracle: JS Number cannot represent the Q14 determinant.
// Triangle coordinates use the equilateral metric x²+y²+xy, not the metric
// of the right triangle used to store barycentric coordinates.
export function incircleDeterminant(a,b,c,d,square) {
  const delta=p=>[BigInt(p[0]-d[0]),BigInt(p[1]-d[1])];
  const [ax,ay]=delta(a),[bx,by]=delta(b),[cx,cy]=delta(c);
  const lift=(x,y)=>x*x+y*y+(square?0n:x*y);
  return lift(ax,ay)*(bx*cy-by*cx)
    -lift(bx,by)*(ax*cy-ay*cx)+lift(cx,cy)*(ax*by-ay*bx);
}

// First establish a valid disk with the requested constrained boundary, then
// check every interior edge. Boundary segments are constrained, never flipped.
// Cocircular ties are valid Delaunay; generator-specific tie ordering is not
// part of this independent geometric contract.
export function auditDelaunay(tile,square,key) {
  audit(tile,square,key);
  const {points,triangles,words}=tile;
  const index=i=>(words[points+(i>>1)]>>>(16*(i&1)))&65535;
  const point=i=>[words[i]&65535,words[i]>>>16];
  const pending=new Map();
  let interiorEdges=0,cocircularEdges=0;
  for(let f=0;f<triangles;f++) {
    const ids=[index(3*f),index(3*f+1),index(3*f+2)];
    for(let e=0;e<3;e++) {
      const a=ids[e],b=ids[(e+1)%3],c=ids[(e+2)%3],id=edgeId(a,b);
      const opposite=pending.get(id);
      if(opposite===undefined){pending.set(id,c);continue;}
      pending.delete(id);
      const determinant=incircleDeterminant(point(a),point(b),point(c),point(opposite),square);
      if(determinant>0n)throw Error(`non-Delaunay edge ${a}:${b}, opposite ${c}:${opposite}, determinant ${determinant}`);
      interiorEdges++;if(determinant===0n)cocircularEdges++;
    }
  }
  return {interiorEdges,cocircularEdges};
}

export function audit(tile,square,key=null) {
  const {points,triangles,words,status}=tile;
  if(status!==0||!(words instanceof Uint32Array))throw Error('failed typed payload');
  if(!Number.isInteger(points)||points<3||points>65536
    ||!Number.isInteger(triangles)||triangles<1||triangles>131072)throw Error('counts');
  if(words.length!==points+Math.ceil(triangles*3/2))throw Error('layout');
  if(triangles%2===1&&words.at(-1)>>>16!==0)throw Error('index padding');
  const coordinates=new Map();
  for(let i=0;i<points;i++) {
    const w=words[i],x=w&65535,y=w>>>16;
    if(x>Q||y>Q||(!square&&x+y>Q))throw Error('coordinate');
    if(coordinates.has(w))throw Error('duplicate point');
    coordinates.set(w,i);
  }
  const expected=new Set();
  if(key) {
    const exponents=square?[key.a,key.b,key.c,key.d]:[key.c,key.a,key.b];
    if(exponents.some(e=>!Number.isInteger(e)||e<0||e>8))throw Error('edge key');
    const corners=square?[[0,0],[Q,0],[Q,Q],[0,Q]]:[[0,0],[Q,0],[0,Q]];
    for(let e=0;e<corners.length;e++) {
      const from=corners[e],to=corners[(e+1)%corners.length],n=2**exponents[e];
      let previous;
      for(let j=0;j<=n;j++) {
        const w=pointWord(from[0]+(to[0]-from[0])*j/n,from[1]+(to[1]-from[1])*j/n);
        const current=coordinates.get(w);
        if(current===undefined)throw Error(`missing boundary point on edge ${e}, step ${j}/${n}`);
        if(previous!==undefined)expected.add(edgeId(previous,current));
        previous=current;
      }
    }
  }
  const index=i=>(words[points+(i>>1)]>>>(16*(i&1)))&65535;
  const edges=new Map(),used=new Uint8Array(points);
  let area=0;
  for(let f=0;f<triangles;f++) {
    const ids=[index(f*3),index(f*3+1),index(f*3+2)];
    if(ids.some(i=>i>=points))throw Error('index');
    const p=ids.map(i=>{used[i]=1;return [words[i]&65535,words[i]>>>16]});
    const cross=(p[1][0]-p[0][0])*(p[2][1]-p[0][1])-(p[1][1]-p[0][1])*(p[2][0]-p[0][0]);
    if(cross<=0)throw Error('winding');
    area+=cross;
    for(let e=0;e<3;e++) {
      const a=ids[e],b=ids[(e+1)%3],id=edgeId(a,b),sign=a<b?1:-1;
      const previous=edges.get(id);
      if(previous===undefined)edges.set(id,sign);
      else if(previous===0||previous===sign)throw Error('edge incidence');
      else edges.set(id,0);
    }
  }
  if(used.some(v=>!v)||area!==(square?2*Q*Q:Q*Q))throw Error('coverage receipt');
  if(points-edges.size+triangles!==1)throw Error('Euler disk');
  if(key) {
    for(const [id,sign] of edges) {
      if(sign!==0&&!expected.delete(id))throw Error('unexpected boundary segment');
      if(sign===0&&expected.has(id))throw Error('boundary segment is interior');
    }
    if(expected.size)throw Error('missing boundary segment');
  }
  return area;
}
