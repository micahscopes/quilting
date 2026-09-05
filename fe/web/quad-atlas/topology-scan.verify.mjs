// Diagnostic only. The browser runs unmodified compiler-emitted scan passes
// on separate buffers, never mutating the demo's resident geometry.
const INVALID=0xffffffff;

export function scanLayout(points,faces) {
  return {points,faces,owner:points*7,expansion:points*7+faces,
    offset:points*7+faces*2,progress:points*7+faces*3,words:points*7+faces*3+8};
}

// Independent ordered oracle, including the deliberately order-sensitive
// "skip an expansion if it would exceed capacity" failure accounting.
export function referenceTopologyScan(input,l) {
  const w=input.slice(), g=l.progress, n=w[g+2], pending=w[g+3];
  if(w[g+6] || w[g+7])return w;
  if(!pending){w[g+4]=n;w[g+5]=0;return w;}
  const proposal=(p,k)=>w[l.points*2+p*4+k];
  const winner=p=>w[l.points*6+p];
  let output=0, invariants=0, locations=0, winners=0, expected=n;
  for(let f=0;f<n;++f){
    w[l.offset+f]=output;
    const e=w[l.expansion+f],p=w[l.owner+f];
    if(e===1){if(p!==INVALID)++invariants;}
    else if(p>=l.points)++invariants;
    else {
      const k=proposal(p,0), target=k===1?3:k===2?2:0;
      const owns=proposal(p,1)===f || (k===2 && proposal(p,2)===f);
      if(e!==target || !winner(p) || !owns)++invariants;
    }
    if(e===0 || e>3)++invariants;
    else if(output+e>l.faces)++invariants;
    else output+=e;
  }
  for(let p=0;p<l.points;++p){
    if(w[(w[g+1]===0?0:l.points)+p]!==0)continue;
    const k=proposal(p,0);
    if(k!==1 && k!==2)++locations;
    if(!winner(p))continue;
    ++winners;
    const first=proposal(p,1),second=proposal(p,2),boundary=k===2 && second===INVALID;
    const count=k===2 && !boundary?2:1;
    expected+=boundary?1:2;
    let owns=Number(first<n && w[l.owner+first]===p);
    if(count===2 && second<n && second!==first && w[l.owner+second]===p)++owns;
    if(owns!==count)++invariants;
  }
  if(winners>pending)++invariants;
  if(winners===0 && locations===0)++invariants;
  if(output!==expected || output>l.faces)++invariants;
  w[g+4]=output;w[g+5]=winners;w[g+6]=locations;w[g+7]=invariants;
  return w;
}

export function topologyScanFixtures(l) {
  const cases=[];
  const make=(name,n,kind=1,parity=0)=>{
    const w=Array(l.words).fill(0),p=l.points-1,g=l.progress;
    w.fill(1,0,l.points*2); // All unused points are already inactive.
    w.fill(INVALID,l.owner,l.owner+l.faces);
    w.fill(1,l.expansion,l.expansion+n);
    w.fill(0x12345678,l.offset,l.offset+l.faces);
    w[g+1]=parity;w[g+2]=n;w[g+3]=1;
    w[parity*l.points+p]=0;
    w[l.points*6+p]=1;
    w[l.points*2+p*4]=kind;
    w[l.points*2+p*4+1]=0;
    w[l.points*2+p*4+2]=kind===3?n-1:INVALID;
    w[l.owner]=p;w[l.expansion]=kind===1?3:2;
    if(kind===3){w[l.points*2+p*4]=2;w[l.owner+n-1]=p;w[l.expansion+n-1]=2;}
    return {name,input:w};
  };
  for(const n of [...new Set([2,63,64,65,l.faces-2])].filter(n=>n>1 && n<=l.faces-2)){
    for(const kind of [1,2,3])for(const parity of [0,1])
      cases.push(make(`valid-${n}-${kind}-${parity}`,n,kind,parity));
  }
  const mutate=(name,fn)=>{const c=make(name,Math.min(65,l.faces-2));fn(c.input);cases.push(c);};
  mutate('unknown-owner',w=>w[l.owner]=INVALID);
  mutate('unchanged-face-with-owner',w=>w[l.owner+1]=l.points-1);
  mutate('missing-location',w=>w[l.points*2+(l.points-1)*4]=0);
  mutate('unclaimed-face',w=>w[l.points*6+l.points-1]=0);
  mutate('wrong-first-face',w=>w[l.points*2+(l.points-1)*4+1]=2);
  mutate('zero-expansion',w=>w[l.expansion]=0);
  mutate('huge-expansion',w=>w[l.expansion+1]=INVALID);
  mutate('wrong-expansion',w=>w[l.expansion]=2);
  mutate('pending-zero',w=>w[l.progress+3]=0);
  mutate('prior-location-failure',w=>w[l.progress+6]=7);
  mutate('prior-invariant-failure',w=>w[l.progress+7]=9);
  mutate('winner-not-pending',w=>w[l.points-1]=1);
  mutate('second-pending-without-location',w=>{w[0]=0;w[l.progress+3]=2;});
  mutate('too-many-winners',w=>{
    w[0]=0;w[l.points*6]=1;w[l.points*2]=1;w[l.points*2+1]=1;
    w[l.points*2+2]=INVALID;w[l.owner+1]=0;w[l.expansion+1]=3;
  });
  cases.push(make('capacity-overflow',l.faces));
  const duplicate=make('duplicate-interior-face',65,3);
  duplicate.input[l.points*2+(l.points-1)*4+2]=0;cases.push(duplicate);
  return cases;
}

export async function verifyTopologyScan(device,manifest,manifestURL) {
  const length=name=>manifest.resources.find(r=>r.name===name).length;
  const l=scanLayout(length('points')/3,length('triangles')/4);
  if(l.words!==length('topology'))throw Error('topology layout mismatch');
  const buffers=[],byName=new Map(),prepared=[];
  const allocate=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
  device.pushErrorScope('validation');let scope=true;
  try {
    for(const name of ['topology','topology_blocks'])byName.set(name,allocate(length(name)*4,
      GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST));
    const stages=['topology_rank','topology_scan','topology_offsets'];
    const shaders=[];
    for(const name of stages){
      const p=manifest.passes.find(p=>p.source_entry===name);
      if(!p || !p.dispatch)throw Error(`missing compiled stage ${name}`);
      const response=await fetch(new URL(p.shader,manifestURL));
      if(!response.ok)throw Error(`shader HTTP ${response.status}`);
      const code=await response.text();shaders.push({name,shader:p.shader,bytes:new TextEncoder().encode(code).length});
      const module=device.createShaderModule({code});
      const info=await module.getCompilationInfo();
      if(info.messages.some(m=>m.type==='error'))throw Error(JSON.stringify(info.messages));
      const layout=device.createBindGroupLayout({entries:p.layout.bindings.map(b=>({
        binding:b.binding,visibility:GPUShaderStage.COMPUTE,
        buffer:{type:b.access==='read'?'read-only-storage':'storage'},
      }))});
      const pipeline=await device.createComputePipelineAsync({
        layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),
        compute:{module,entryPoint:p.layout.entry_point},
      });
      const entries=p.layout.bindings.map(b=>{
        let buffer=byName.get(b.name);
        if(!buffer && (b.role==='input'||b.role==='output'))buffer=allocate(Math.max(4,b.span),GPUBufferUsage.STORAGE);
        if(!buffer || b.group!==0)throw Error(`unexpected scan binding ${b.name}`);
        return {binding:b.binding,resource:{buffer}};
      });
      prepared.push({pipeline,group:device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries}),dispatch:p.dispatch});
    }
    const staging=allocate(l.words*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const observations=[];
    for(const c of topologyScanFixtures(l)){
      const expected=referenceTopologyScan(c.input,l);
      device.queue.writeBuffer(byName.get('topology'),0,new Uint32Array(c.input));
      // Poison reused summaries: inactive blocks must not influence this job.
      device.queue.writeBuffer(byName.get('topology_blocks'),0,new Uint32Array(length('topology_blocks')).fill(INVALID));
      const enc=device.createCommandEncoder();
      for(const p of prepared){const pass=enc.beginComputePass();pass.setPipeline(p.pipeline);pass.setBindGroup(0,p.group);pass.dispatchWorkgroups(...p.dispatch);pass.end();}
      enc.copyBufferToBuffer(byName.get('topology'),0,staging,0,l.words*4);
      device.queue.submit([enc.finish()]);await staging.mapAsync(GPUMapMode.READ);
      const actual=[...new Uint32Array(staging.getMappedRange())];staging.unmap();
      const mismatch=actual.findIndex((v,i)=>v!==expected[i]);
      if(mismatch!==-1){
        const error=await device.popErrorScope();scope=false;
        throw Error(error?.message ?? `${c.name}: word ${mismatch}: GPU=${actual[mismatch]} reference=${expected[mismatch]}`);
      }
      observations.push({name:c.name,progress:actual.slice(l.progress),comparedWords:l.words});
    }
    const error=await device.popErrorScope();scope=false;if(error)throw Error(error.message);
    return {manifest:manifestURL,passes:manifest.passes.length,shaders,layout:l,observations};
  } finally {for(const b of buffers)b.destroy();if(scope)await device.popErrorScope();}
}
