// Test instrumentation only: Fe owns job scheduling, retention and cancellation.
import {createStructuredWorkerScopes,createStructuredWorkerMailboxes,
  createMaterializedTaskRegistry} from './generated-pool/tasks.js';
import {createHostCompletionBroker} from './generated-pool/host-completion.js';
import {audit} from './packed-audit.mjs';

export async function verifyAtlasPool({cancelAfterMs=null,onProgress=()=>{},onMemoryEvent=null}={}) {
  const start=performance.now();
  const original=await createStructuredWorkerScopes();
  const receipts=[];
  let active=0,peakActive=0;
  const scopes=original.map((item,index)=>({...item,scope:{
    spawn:(...args)=>item.scope.spawn(...args),
    failure:(...args)=>item.scope.failure(...args),
    close:(...args)=>item.scope.close(...args),
    async request(lane,payload,signal) {
      const begin=performance.now();
      active++; peakActive=Math.max(peakActive,active);
      try {
        const response=await item.scope.request(lane,payload,signal);
        const roundtripMs=performance.now()-begin;
        if(response.lease.epoch!==payload.lease.epoch||response.lease.ordinal!==payload.lease.ordinal
          ||response.lease.slot!==payload.lease.slot) throw Error('lease mismatch');
        const auditStart=performance.now();
        audit(response.tile,payload.key.square,payload.key);
        receipts.push({worker:index,ordinal:payload.lease.ordinal,key:payload.key,
          points:response.tile.points,triangles:response.tile.triangles,
          bytes:response.tile.words.byteLength,roundtripMs,auditMs:performance.now()-auditStart});
        return response;
      } finally {active--;}
    },
  }}));
  let instance;
  const broker=createHostCompletionBroker({workerScopes:scopes,actorEvents:{
    send(event,signal) {
      if(signal.aborted)throw new DOMException('cancelled','AbortError');
      instance.exports.fe_actor_transition_v1(...event);
    },
  }});
  const mailbox=createStructuredWorkerMailboxes(scopes,broker.completions);
  ({instance}=await WebAssembly.instantiateStreaming(fetch('./generated-pool/parent.wasm'),
    Object.assign({},broker.imports,mailbox)));
  // Optional bounded test observation of the real allocator; never replaces
  // allocation, release, task execution or scheduling behavior.
  const observedExports={...instance.exports};
  if(onMemoryEvent)for(const name of ['cabi_realloc','fe_cabi_post_return']) {
    const original=instance.exports[name];
    observedExports[name]=(...args)=>{
      const before=instance.exports.fe_cabi_checkpoint?.();
      try {
        const result=original(...args);
        onMemoryEvent({name,args,before,after:instance.exports.fe_cabi_checkpoint?.(),result});
        return result;
      }catch(error){
        onMemoryEvent({name,args,before,after:instance.exports.fe_cabi_checkpoint?.(),error:String(error)});
        throw error;
      }
    };
  }
  mailbox.attach(onMemoryEvent?observedExports:instance.exports);
  const initial=instance.exports.fe_actor_initialize_v1();
  const tasks=createMaterializedTaskRegistry(instance.exports);
  const controller=new AbortController();
  const setupMs=performance.now()-start;
  const allTasks=Object.values(tasks);
  const supervisors=allTasks.filter(task=>task.inputWidth===0)
    .map(task=>broker.run(task,[],{signal:controller.signal}));
  const settledSupervisors=Promise.allSettled(supervisors);
  const read=()=>{
    const [address,revision,epoch,total,completed,words,points,triangles,status]=instance.exports.fe_actor_project_v1();
    return {address,revision,epoch,total,completed,words,points,triangles,status,
      linearMemoryBytes:instance.exports.memory.buffer.byteLength,active,peakActive,
      elapsedMs:performance.now()-start};
  };
  const progress=setInterval(()=>onProgress(read()),250);
  const cancellation=cancelAfterMs===null?null:setTimeout(()=>instance.exports.fe_actor_transition_v1(1),cancelAfterMs);
  try {
    const counts=await Promise.all(allTasks.filter(task=>task.inputWidth!==0)
      .map(task=>broker.run(task,initial,{signal:controller.signal})));
    const result=read();
    if(cancelAfterMs===null) {
      if(result.status!==0||result.total!==1200||result.completed!==1200)throw Error(`incomplete atlas: ${JSON.stringify(result)}`);
      if(new Set(receipts.map(r=>r.ordinal)).size!==1200)throw Error('duplicate or missing tile');
      if(receipts.reduce((sum,r)=>sum+r.bytes,0)!==result.words*4)throw Error('retained byte count');
    } else if(result.status!==1)throw Error('cancellation not observed');
    return {...result,setupMs,counts,receipts,cancelled:cancelAfterMs!==null};
  } finally {
    clearInterval(progress); if(cancellation!==null)clearTimeout(cancellation);
    controller.abort(); await settledSupervisors;
  }
}
