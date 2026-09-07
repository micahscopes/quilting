// Browser acceptance driver only. Atlas algorithms and transport policy are
// authored Fe plus the compiler's unmodified actor runtime, not this test.
import {compileActorAdapter,canonicalInterfaceManifest} from './generated/interface.js';
import {createCanonicalModuleWorkerActor} from './generated/runtime/module-worker-actor.js';

import {audit} from './packed-audit.mjs';
export {audit} from './packed-audit.mjs';

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
      const area=audit(response.tile,square,{a:key[0],b:key[1],c:key[2],d:key[3]});
      results.push({square,key,points:response.tile.points,triangles:response.tile.triangles,
        bytes:response.tile.words.byteLength,roundtripMs,auditMs:performance.now()-auditStart,area});
    }
    const invalid=await worker.request(lane,{epoch:5,square:false,a:9,b:9,c:9,d:0,seed:42},5);
    if(invalid.tile.status!==2||invalid.tile.words.length!==0)throw Error('invalid key publication');
    return {state:'passed',compileMs,startupMs,results,invalidKeyRejected:true,workers:1};
  } finally { worker.close(); }
}
