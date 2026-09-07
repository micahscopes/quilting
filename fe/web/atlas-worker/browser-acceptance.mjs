// Browser acceptance driver only. Atlas algorithms and transport policy are
// authored Fe plus the compiler's unmodified actor runtime, not this test.
import {compileActorAdapter,canonicalInterfaceManifest} from './generated/interface.js';
import {createCanonicalModuleWorkerActor} from './generated/runtime/module-worker-actor.js';

export function audit(tile,square) {
  const {points,triangles,words,status}=tile;
  if(status!==0 || !(words instanceof Uint32Array)) throw Error('failed typed payload');
  if(words.length!==points+Math.ceil(triangles*3/2)) throw Error('layout');
  const index=i=>(words[points+(i>>1)]>>>(16*(i&1)))&65535;
  let area=0;
  const used=new Uint8Array(points);
  for(let f=0;f<triangles;f++) {
    const ids=[index(f*3),index(f*3+1),index(f*3+2)];
    if(ids.some(i=>i>=points)) throw Error('index');
    const p=ids.map(i=>{used[i]=1;return [words[i]&65535,words[i]>>>16]});
    if(p.some(([x,y])=>x>16384||y>16384)) throw Error('coordinate');
    const cross=(p[1][0]-p[0][0])*(p[2][1]-p[0][1])-(p[1][1]-p[0][1])*(p[2][0]-p[0][0]);
    if(cross<=0) throw Error('winding');
    area+=cross;
  }
  if(used.some(v=>!v)||area!==(square?536870912:268435456)) throw Error('coverage receipt');
  return area;
}

export async function verifyAtlasWorker() {
  const compileStart=performance.now();
  const wasm=await WebAssembly.compileStreaming(fetch(new URL('./generated/child.wasm',import.meta.url)));
  const compileMs=performance.now()-compileStart;
  const start=performance.now();
  const worker=await createCanonicalModuleWorkerActor({
    workerUrl:new URL('./generated/runtime/worker-host.js',import.meta.url),
    init:{wasm},adapter:compileActorAdapter(),maxPending:2,
  });
  const startupMs=performance.now()-start;
  const lane=canonicalInterfaceManifest.lanes[0].name;
  const results=[];
  try {
    for(const square of [false,true]) for(const key of [[0,0,8,0],[8,8,8,8]]) {
      const epoch=results.length+1;
      const requestStart=performance.now();
      const response=await worker.request(lane,{epoch,square,a:key[0],b:key[1],c:key[2],d:key[3],seed:42},epoch);
      const roundtripMs=performance.now()-requestStart;
      if(response.epoch!==epoch) throw Error('epoch');
      const auditStart=performance.now();
      const area=audit(response.tile,square);
      results.push({square,key,points:response.tile.points,triangles:response.tile.triangles,
        bytes:response.tile.words.byteLength,roundtripMs,auditMs:performance.now()-auditStart,area});
    }
    const invalid=await worker.request(lane,{epoch:5,square:false,a:9,b:9,c:9,d:0,seed:42},5);
    if(invalid.tile.status!==2||invalid.tile.words.length!==0)throw Error('invalid key publication');
    return {state:'passed',compileMs,startupMs,results,invalidKeyRejected:true,workers:1};
  } finally { worker.close(); }
}
