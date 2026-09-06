// Diagnostic only: replay compiled Fe passes into private buffers. No geometry
// generator, shader replacement, production scheduling, or downloaded atlas.
// The full-atlas outer cycle is replaced by one recorded job; inner cycles
// retain their compiled order. Repair is observed before its final certificate.
export function tileReplayBlocks(passes) {
  const start = passes.findIndex(p => p.source_entry === 'initialize');
  const end = passes.findIndex(p => p.source_entry === 'repair_finish');
  if (start < 0 || end <= start) throw Error('missing full-atlas tile interval');
  const outer = passes[start].cycle;
  if (!outer || outer.inner) throw Error('unexpected tile initialization cycle');
  const blocks = [];
  const seen = new Set();
  for (let i = start; i <= end;) {
    const p = passes[i], cycle = p.cycle;
    if (cycle?.group !== outer.group || cycle.repeat !== outer.repeat)
      throw Error('tile pass escapes outer atlas cycle');
    const inner = cycle.inner;
    if (!inner) { blocks.push({passes: [p], repeat: 1}); i++; continue; }
    if (inner.inner || seen.has(inner.group)) throw Error('unsupported or noncontiguous inner cycle');
    if (!Number.isSafeInteger(inner.repeat) || inner.repeat < 1) throw Error('invalid cycle count');
    seen.add(inner.group);
    const members = [];
    while (i <= end && passes[i].cycle?.inner?.group === inner.group) {
      const c = passes[i].cycle;
      if (c.group !== outer.group || c.repeat !== outer.repeat || c.inner.repeat !== inner.repeat || c.inner.inner)
        throw Error('inconsistent inner cycle');
      members.push(passes[i++]);
    }
    blocks.push({passes: members, repeat: inner.repeat});
  }
  if (blocks.filter(b => b.passes.some(p => p.source_entry === 'repair_apply')).length !== 1)
    throw Error('missing repair block');
  return blocks;
}

// A diagnostic prefix ends at a complete compiled cycle, never halfway through
// an iteration. This is not an alternative production scheduler.
export function tileReplayPrefix(passes, stopAfter) {
  const blocks = tileReplayBlocks(passes);
  const end = blocks.findIndex(b => b.passes.at(-1).source_entry === stopAfter);
  if (end < 0) throw Error('prefix must end at a compiled block boundary');
  return blocks.slice(0, end + 1);
}

export async function diagnoseAtlasPrefix(gpu, manifest, activeWords,
  {stopAfter = 'initialize', passesPerSubmission = 1, onProgress = () => {}, inspectSampling} = {}) {
  if (![7,10].includes(activeWords.length) || !activeWords.every(v => Number.isSafeInteger(v) && v >= 0 && v <= 0xffffffff)
      || activeWords[activeWords.length === 7 ? 4 : 5] !== 1)
    throw Error('expected a recorded valid atlas job');
  if (!Number.isSafeInteger(passesPerSubmission) || passesPerSubmission < 1 || passesPerSubmission > 256)
    throw Error('submission bound must be within 1..256');
  if (inspectSampling !== undefined && typeof inspectSampling !== 'function')
    throw Error('sampling inspection must be a diagnostic callback');
  const blocks = tileReplayPrefix(manifest.passes, stopAfter);
  const device = gpu.device, owned = [], resources = new Map(), prepared = new Map(), outputs = [];
  let allocatedBytes = 0;
  const make = (size, usage) => {
    if (!Number.isSafeInteger(size) || size < 4 || size % 4 || allocatedBytes + size > 512*1024*1024)
      throw Error('prefix allocation exceeds diagnostic bounds');
    const buffer = device.createBuffer({size,usage}); owned.push(buffer); allocatedBytes += size; return buffer;
  };
  const needed = new Set(blocks.flatMap(b=>b.passes.flatMap(p=>p.layout.bindings.filter(b=>b.role==='resource').map(b=>b.name))));
  device.pushErrorScope('validation'); let scopeOpen = true;
  try {
    for (const r of manifest.resources.filter(r=>needed.has(r.name))) {
      if (r.artifact) throw Error('prefix must generate its own data');
      resources.set(r.name,make(r.length*r.stride,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC
        |(r.buffer_usage?.includes('indirect')?GPUBufferUsage.INDIRECT:0)));
    }
    if (resources.get('active')?.size !== activeWords.length*4) throw Error('recorded job layout mismatch');
    device.queue.writeBuffer(resources.get('active'),0,new Uint32Array(activeWords));
    for (const block of blocks) for (const p of block.passes) {
      if (p.layout.graph_failure || (p.repeat !== undefined && p.repeat !== 1))
        throw Error('prefix does not own graph failure epochs or per-pass repeats');
      const record = gpu.passRecords.find(r=>r.pass.source_entry===p.source_entry);
      const pipeline = record?.pipeline ?? await record?.pipelinePromise;
      if (!pipeline) throw Error(`missing compiled pipeline ${p.source_entry}`);
      const entries = p.layout.bindings.map(b=>{
        if (b.group !== 0) throw Error('unsupported diagnostic binding group');
        if (b.role==='resource' && resources.has(b.name))
          return {binding:b.binding,resource:{buffer:resources.get(b.name)}};
        if (b.role==='output' && b.name==='trap' && b.stride===4 && b.access==='read_write') {
          const buffer=make(b.span,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
          outputs.push({entry:p.source_entry,buffer});
          return {binding:b.binding,resource:{buffer}};
        }
        throw Error(`unsupported prefix binding ${b.name}`);
      });
      prepared.set(p.source_entry,{pipeline,group:device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries})});
    }
    const batches=[]; let encoder=device.createCommandEncoder(), names=[];
    const flush=async()=>{
      if (!names.length) return;
      const row={passes:[...names],status:'submitted',started:performance.now()}; batches.push(row); onProgress({...row});
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      row.status='completed'; row.elapsedMs=performance.now()-row.started; onProgress({...row});
      encoder=device.createCommandEncoder(); names=[];
    };
    for (const block of blocks) for (let n=0;n<block.repeat;n++) for (const p of block.passes) {
      const r=prepared.get(p.source_entry), pass=encoder.beginComputePass();
      pass.setPipeline(r.pipeline); pass.setBindGroup(0,r.group);
      if (p.dispatch_indirect) pass.dispatchWorkgroupsIndirect(resources.get(p.dispatch_indirect.resource),p.dispatch_indirect.offset??0);
      else pass.dispatchWorkgroups(...p.dispatch);
      pass.end(); names.push(p.source_entry);
      if (names.length===passesPerSubmission) await flush();
    }
    await flush();
    const lastInvocationTraps=[];
    for (const {entry,buffer} of outputs) {
      const mapped=make(buffer.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      const copy=device.createCommandEncoder(); copy.copyBufferToBuffer(buffer,0,mapped,0,buffer.size);
      device.queue.submit([copy.finish()]); await mapped.mapAsync(GPUMapMode.READ);
      const words=new Uint32Array(mapped.getMappedRange()); let nonzero=0;
      for (const word of words) if (word) nonzero++;
      lastInvocationTraps.push({entry,words:words.length,nonzero}); mapped.unmap();
    }
    const stateSnapshots={};
    for (const name of ['active','receipt','repair_state']) {
      const buffer=resources.get(name);
      if (!buffer) continue;
      if (buffer.size>4096) throw Error('state snapshot exceeds diagnostic bounds');
      const mapped=make(buffer.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      const copy=device.createCommandEncoder(); copy.copyBufferToBuffer(buffer,0,mapped,0,buffer.size);
      device.queue.submit([copy.finish()]); await mapped.mapAsync(GPUMapMode.READ);
      stateSnapshots[name]=Array.from(new Uint32Array(mapped.getMappedRange())); mapped.unmap();
    }
    // Optional offline observation, never an input to subsequent GPU work.
    // The callback returns a compact census, not megabytes of raw snapshot JSON.
    let samplingInspection;
    if (inspectSampling) {
      const buffer=resources.get('sampling');
      if (!buffer || buffer.size>32*1024*1024) throw Error('sampling inspection exceeds diagnostic bounds');
      const mapped=make(buffer.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      const copy=device.createCommandEncoder(); copy.copyBufferToBuffer(buffer,0,mapped,0,buffer.size);
      device.queue.submit([copy.finish()]); await mapped.mapAsync(GPUMapMode.READ);
      try { samplingInspection=await inspectSampling(new Uint32Array(mapped.getMappedRange())); }
      finally { mapped.unmap(); }
    }
    const error=await device.popErrorScope(); scopeOpen=false;
    if (error) throw Error(error.message);
    return {stopAfter,passesPerSubmission,batches,allocatedBytes,lastInvocationTraps,stateSnapshots,
      ...(inspectSampling?{samplingInspection}:{}),
      trapCoverage:'last stored invocation slots only; not a whole-epoch certificate'};
  } finally {
    try { if (scopeOpen) await device.popErrorScope(); }
    finally { for (const buffer of owned) buffer.destroy(); }
  }
}

export async function replayAtlasTile(gpu, manifest, activeWords, {rounds = [128,256,512,1024], captureGeometry = false,
  sampleGroups, captureSampling = false, poisonSampling = false, captureInitialGeometry = false, shape = 'quad'} = {}) {
  if (shape !== 'quad' && shape !== 'triangle') throw Error('unknown atlas shape');
  const keyWords = shape === 'quad' ? 4 : 3;
  if (activeWords.length !== (shape === 'quad' ? 10 : 7)
      || !activeWords.every(x => Number.isSafeInteger(x) && x >= 0 && x <= 0xffffffff)
      || activeWords[keyWords+1] !== 1) throw Error(`expected a recorded valid ${shape} job`);
  if (!rounds.length || rounds.some((n,i) => !Number.isSafeInteger(n) || n < 1 || n > 65535 || (i > 0 && n <= rounds[i-1])))
    throw Error('repair checkpoints must increase within 1..65535');
  if (sampleGroups !== undefined && (!Number.isSafeInteger(sampleGroups) || sampleGroups < 1 || sampleGroups > 65535))
    throw Error('invalid diagnostic sampling dispatch');
  const blocks = tileReplayBlocks(manifest.passes);
  const device = gpu.device, owned = [], resources = new Map(), prepared = new Map();
  const make = (size,usage) => { const b = device.createBuffer({size,usage}); owned.push(b); return b; };
  const needed = new Set(blocks.flatMap(b => b.passes.flatMap(p => p.layout.bindings.map(b => b.name))));
  device.pushErrorScope('validation'); let scopeOpen = true;
  try {
    for (const r of manifest.resources.filter(r => needed.has(r.name))) {
      if (r.artifact) throw Error('diagnostic must generate its own geometry');
      resources.set(r.name, make(r.length*r.stride, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        | (r.buffer_usage?.includes('indirect') ? GPUBufferUsage.INDIRECT : 0)));
    }
    const active = resources.get('active');
    if (active?.size !== activeWords.length*4) throw Error('recorded job layout mismatch');
    device.queue.writeBuffer(active,0,new Uint32Array(activeWords));
    if (poisonSampling) {
      const sampling=resources.get('sampling');
      if (sampling.size > 32*1024*1024) throw Error('sampling poison exceeds diagnostic bounds');
      device.queue.writeBuffer(sampling,0,new Uint32Array(sampling.size/4).fill(0xdeadbeef));
    }
    for (const block of blocks) for (const p of block.passes) {
      if (p.repeat !== undefined && p.repeat !== 1) throw Error('unsupported per-pass repetition');
      const record = gpu.passRecords.find(r => r.pass.source_entry === p.source_entry);
      const pipeline = record?.pipeline ?? await record?.pipelinePromise;
      if (!pipeline) throw Error(`missing compiled pipeline ${p.source_entry}`);
      const entries = p.layout.bindings.map(b => {
        if (b.role !== 'resource' || b.group !== 0 || !resources.has(b.name)) throw Error(`unsupported binding ${b.name}`);
        return {binding:b.binding,resource:{buffer:resources.get(b.name)}};
      });
      prepared.set(p.source_entry,{pipeline,group:device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries})});
    }
    const state = resources.get('repair_state'), receipt = resources.get('receipt');
    const mapped = make(state.size+receipt.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const observations = [];
    let encoder = device.createCommandEncoder();
    const dispatch = p => {
      const r = prepared.get(p.source_entry), pass = encoder.beginComputePass();
      pass.setPipeline(r.pipeline); pass.setBindGroup(0,r.group);
      if (sampleGroups !== undefined && ['propose','retire'].includes(p.source_entry)) {
        if (p.dispatch_indirect || sampleGroups > p.dispatch[0]) throw Error('override requires the fixed-dispatch baseline');
        pass.dispatchWorkgroups(sampleGroups,p.dispatch[1],p.dispatch[2]);
      } else if (p.dispatch_indirect) pass.dispatchWorkgroupsIndirect(resources.get(p.dispatch_indirect.resource),p.dispatch_indirect.offset??0);
      else pass.dispatchWorkgroups(...p.dispatch);
      pass.end();
    };
    const observe = async label => {
      encoder.copyBufferToBuffer(state,0,mapped,0,state.size);
      encoder.copyBufferToBuffer(receipt,0,mapped,state.size,receipt.size);
      device.queue.submit([encoder.finish()]);
      await mapped.mapAsync(GPUMapMode.READ);
      const words = Array.from(new Uint32Array(mapped.getMappedRange())); mapped.unmap();
      observations.push({label,state:words.slice(0,state.size/4),receipt:words.slice(state.size/4)});
      encoder = device.createCommandEncoder();
    };
    const readGeometry = async () => {
      const last = observations.at(-1), [nv,nt] = last.state;
      const vertexBytes = nv*12, triangleBytes = nt*16;
      if (!nv || !nt || vertexBytes+triangleBytes > 4*1024*1024
          || vertexBytes > resources.get('points').size || triangleBytes > resources.get('triangles').size)
        throw Error('geometry observation exceeds diagnostic bounds');
      const geometryRead = make(vertexBytes+triangleBytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      encoder.copyBufferToBuffer(resources.get('points'),0,geometryRead,0,vertexBytes);
      encoder.copyBufferToBuffer(resources.get('triangles'),0,geometryRead,vertexBytes,triangleBytes);
      device.queue.submit([encoder.finish()]);
      await geometryRead.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(geometryRead.getMappedRange());
      // Match each Fe PlanarCoordinates instance: quad stores x,y,padding;
      // triangle stores barycentric a,b,c and its planar chart is (b,c).
      const coordinateOffset = shape === 'triangle' ? 1 : 0;
      const result = {key:activeWords.slice(0,keyWords),receipt:last.receipt,
        points:Array.from({length:nv},(_,i)=>Array.from(words.slice(i*3+coordinateOffset,i*3+coordinateOffset+2))),
        triangles:Array.from({length:nt},(_,i)=>Array.from(words.slice(nv*3+i*4,nv*3+i*4+3)))};
      geometryRead.unmap(); encoder = device.createCommandEncoder();
      return result;
    };
    let initialGeometry;
    for (const block of blocks) {
      if (block.passes.some(p => p.source_entry === 'repair_apply')) {
        await observe('before-repair');
        if (captureInitialGeometry) initialGeometry = await readGeometry();
        let done = 0;
        for (const n of rounds) {
          for (; done < n; done++) for (const p of block.passes) dispatch(p);
          await observe(`after-${n}-rounds`);
        }
      } else for (let n = 0; n < block.repeat; n++) for (const p of block.passes) dispatch(p);
    }
    await observe('final-certificate');
    let samplingSha256;
    if (captureSampling) {
      const source = resources.get('sampling');
      if (source.size > 32*1024*1024) throw Error('sampling observation exceeds diagnostic bounds');
      const read = make(source.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      encoder.copyBufferToBuffer(source,0,read,0,source.size);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const hash = await crypto.subtle.digest('SHA-256',read.getMappedRange());
      samplingSha256 = Array.from(new Uint8Array(hash),v=>v.toString(16).padStart(2,'0')).join('');
      read.unmap(); encoder = device.createCommandEncoder();
    }
    const geometry = captureGeometry ? await readGeometry() : undefined;
    const error = await device.popErrorScope(); scopeOpen = false;
    if (error) throw Error(error.message);
    return {shape,activeWords,rounds,observations,...(poisonSampling?{poisonSampling:true}:{}),...(sampleGroups!==undefined?{sampleGroups}:{}),
      ...(samplingSha256?{samplingSha256}:{}),...(geometry?{geometry}:{}),...(initialGeometry?{initialGeometry}:{}),
      allocatedBytes:[...resources.values()].reduce((sum,b)=>sum+b.size,0)};
  } finally {
    if (scopeOpen) await device.popErrorScope();
    for (const b of owned) b.destroy();
  }
}
