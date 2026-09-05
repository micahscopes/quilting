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

export async function replayAtlasTile(gpu, manifest, activeWords, {rounds = [128,256,512,1024]} = {}) {
  if (activeWords.length !== 10 || !activeWords.every(x => Number.isSafeInteger(x) && x >= 0 && x <= 0xffffffff)
      || activeWords[5] !== 1) throw Error('expected a recorded valid quad job');
  if (!rounds.length || rounds.some((n,i) => !Number.isSafeInteger(n) || n < 1 || n > 65535 || (i > 0 && n <= rounds[i-1])))
    throw Error('repair checkpoints must increase within 1..65535');
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
      if (p.dispatch_indirect) pass.dispatchWorkgroupsIndirect(resources.get(p.dispatch_indirect.resource),p.dispatch_indirect.offset??0);
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
    for (const block of blocks) {
      if (block.passes.some(p => p.source_entry === 'repair_apply')) {
        await observe('before-repair');
        let done = 0;
        for (const n of rounds) {
          for (; done < n; done++) for (const p of block.passes) dispatch(p);
          await observe(`after-${n}-rounds`);
        }
      } else for (let n = 0; n < block.repeat; n++) for (const p of block.passes) dispatch(p);
    }
    await observe('final-certificate');
    const error = await device.popErrorScope(); scopeOpen = false;
    if (error) throw Error(error.message);
    return {activeWords,rounds,observations,allocatedBytes:[...resources.values()].reduce((sum,b)=>sum+b.size,0)};
  } finally {
    if (scopeOpen) await device.popErrorScope();
    for (const b of owned) b.destroy();
  }
}
