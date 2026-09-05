// Diagnostic replay of the *compiled* actor pipelines into separate buffers.
// Normally only the repair-cycle repetition count changes. An explicit test
// option corrupts one generated face before index validation. Production
// shaders, resource layouts and the final certificate remain unmodified.
export async function verifyRepairBudgets(gpu,manifest,params,options={}){
  const {captureIndex=false,corruptFace}=options;
  const buffers=[],resources=new Map();
  const allocate=(size,usage)=>{const b=gpu.device.createBuffer({size,usage});buffers.push(b);return b;};
  const device=gpu.device;
  device.pushErrorScope('validation');let scope=true;
  try {
    for(const r of manifest.resources){
      if(r.artifact)throw Error('budget test must generate its own geometry');
      resources.set(r.name,allocate(r.length*r.stride,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST));
    }
    const prepared=[];
    for(const record of gpu.passRecords){
      const p=record.pass;if(p.layout.mode!=='compute')continue;
      const pipeline=record.pipeline??await record.pipelinePromise;
      if(!pipeline)throw Error(`pipeline not ready: ${p.source_entry}`);
      const entries=p.layout.bindings.map(b=>{
        let buffer=resources.get(b.name);
        if(b.role==='input'){
          buffer=allocate(Math.max(4,b.span),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
          const bytes=new ArrayBuffer(Math.max(4,b.span)),view=new DataView(bytes);
          for(const member of b.members){
            if(member.scalar!=='f32' || !(member.name in params))throw Error(`unknown input ${member.name}`);
            view.setFloat32(member.offset,params[member.name],true);
          }
          device.queue.writeBuffer(buffer,0,bytes);
        }else if(b.role==='output')buffer=allocate(Math.max(4,b.span),GPUBufferUsage.STORAGE);
        if(!buffer || b.group!==0)throw Error(`unsupported binding ${b.name}`);
        return {binding:b.binding,resource:{buffer}};
      });
      prepared.push({p,pipeline,group:device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries})});
    }
    const repairGroup=prepared.find(r=>r.p.source_entry==='repair_apply')?.p.cycle?.group;
    if(repairGroup===undefined)throw Error('missing repair cycle');
    const source=resources.get('repair_state'),receipt=resources.get('receipt'),draw=resources.get('draw');
    const geometryNames=['points','triangles','edge_index'];
    const geometryBytes=captureIndex?geometryNames.reduce((n,key)=>n+resources.get(key).size,0):0;
    const staging=allocate(source.size+receipt.size+draw.size+geometryBytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const observations=[];
    for(const budget of (corruptFace?[64]:[0,1,64])){
      let encoder=device.createCommandEncoder();
      const dispatch=r=>{
        const pass=encoder.beginComputePass();pass.setPipeline(r.pipeline);pass.setBindGroup(0,r.group);pass.dispatchWorkgroups(...r.p.dispatch);pass.end();
        if(corruptFace && r.p.source_entry==='delaunay_initialize'){
          // Fault injection after generated topology, before parallel validation.
          device.queue.submit([encoder.finish()]);
          device.queue.writeBuffer(resources.get('triangles'),0,new Uint32Array(corruptFace));
          encoder=device.createCommandEncoder();
        }
      };
      for(let i=0;i<prepared.length;){
        const cycle=prepared[i].p.cycle;
        if(!cycle){dispatch(prepared[i++]);continue;}
        let end=i+1;while(end<prepared.length && prepared[end].p.cycle?.group===cycle.group)++end;
        const repeats=cycle.group===repairGroup?budget:cycle.repeat;
        for(let n=0;n<repeats;++n)for(let j=i;j<end;++j)dispatch(prepared[j]);
        i=end;
      }
      encoder.copyBufferToBuffer(source,0,staging,0,source.size);
      encoder.copyBufferToBuffer(receipt,0,staging,source.size,receipt.size);
      encoder.copyBufferToBuffer(draw,0,staging,source.size+receipt.size,draw.size);
      if(captureIndex && budget===0){
        let offset=source.size+receipt.size+draw.size;
        for(const name of geometryNames){const b=resources.get(name);encoder.copyBufferToBuffer(b,0,staging,offset,b.size);offset+=b.size;}
      }
      device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
      const words=[...new Uint32Array(staging.getMappedRange())];staging.unmap();
      const state=words.slice(0,8),r=words.slice(8,24),d=words.slice(24,28);
      if(state[7]!==1 || r[6]!==1)throw Error(`invalid source: ${words}`);
      if(corruptFace){
        if(state[4]===0 || state[5]!==0 || r[10]!==0 || d[0]!==0)throw Error(`invalid face was published: ${words}`);
      }else if(state[4]!==0)throw Error(`invalid repair: ${words}`);
      else if(budget<64){
        if(state[3]===0 || state[5]!==0 || r[10]!==0 || d[0]!==0)
          throw Error(`exhausted budget published an incomplete tile: ${words}`);
      }else if(state[3]!==0 || state[5]!==1 || r[10]!==1 || d[0]!==state[1]*3)
        throw Error(`full budget failed: ${words}`);
      let geometry;
      if(captureIndex && budget===0){
        let offset=28;const data={};
        for(const name of geometryNames){const length=resources.get(name).size/4;data[name]=words.slice(offset,offset+length);offset+=length;}
        const pointCapacity=resources.get('points').size/12;
        geometry={points:Array.from({length:state[0]},(_,i)=>data.points.slice(i*3,i*3+2)),
          triangles:Array.from({length:state[1]},(_,i)=>data.triangles.slice(i*4,i*4+3)),
          index:{offsets:data.edge_index.slice(0,state[0]+1),edges:data.edge_index.slice(pointCapacity+1,pointCapacity+1+state[1]*3)}};
      }
      observations.push({budget,state,receipt:r,draw:d,...(geometry?{geometry}:{})});
    }
    const error=await device.popErrorScope();scope=false;if(error)throw Error(error.message);
    return {params,passes:manifest.passes.length,...(corruptFace?{corruptFace}:{}),observations};
  }finally {for(const b of buffers)b.destroy();if(scope)await device.popErrorScope();}
}
