// Diagnostic execution of the compiled Fe job iterator. Readback buffers are
// private to this test; production resource permissions stay unchanged.
export async function captureQuadJobs(gpu, manifest, realizePassPipeline) {
  const device = gpu.device;
  const owned = [], resources = new Map();
  const allocate = (size, usage) => {
    const buffer = device.createBuffer({size, usage});
    owned.push(buffer);
    return buffer;
  };
  device.pushErrorScope('validation');
  let scopeOpen = true;
  try {
    for (const resource of manifest.resources) {
      if (resource.artifact) throw Error('job test must not import an artifact');
      resources.set(resource.name, allocate(resource.length * resource.stride,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC));
    }
    const stages = [];
    for (const record of gpu.passRecords) {
      const pass = record.pass;
      if (pass.layout.mode !== 'compute' || pass.dispatch_indirect || pass.cycles?.length)
        throw Error('unexpected job fixture schedule');
      const pipeline = await realizePassPipeline(device, record);
      const entries = pass.layout.bindings.map(binding => {
        const buffer = resources.get(binding.name);
        if (!buffer || binding.group !== 0) throw Error(`unexpected binding ${binding.name}`);
        return {binding: binding.binding, resource: {buffer}};
      });
      stages.push({pass, pipeline, group: device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0), entries,
      })});
    }
    if (stages.map(s => s.pass.source_entry).join(',') !==
        'initialize_jobs,next_job,record_job,finish_jobs')
      throw Error('unexpected job fixture stages');
    const cycle = stages[1].pass.cycle;
    if (!cycle || cycle.group !== stages[2].pass.cycle?.group ||
        cycle.repeat !== stages[2].pass.cycle.repeat ||
        cycle.repeat !== resources.get('history').size / 4)
      throw Error('inconsistent canonical job count');
    const history = resources.get('history'), receipt = resources.get('receipt');
    const readback = allocate(history.size + receipt.size,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const encoder = device.createCommandEncoder();
    const dispatch = stage => {
      const pass = encoder.beginComputePass();
      pass.setPipeline(stage.pipeline);
      pass.setBindGroup(0, stage.group);
      pass.dispatchWorkgroups(...stage.pass.dispatch);
      pass.end();
    };
    dispatch(stages[0]);
    for (let job = 0; job < cycle.repeat; job++) {
      dispatch(stages[1]); dispatch(stages[2]);
    }
    dispatch(stages[3]);
    encoder.copyBufferToBuffer(history, 0, readback, 0, history.size);
    encoder.copyBufferToBuffer(receipt, 0, readback, history.size, receipt.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const error = await device.popErrorScope();
    scopeOpen = false;
    if (error) throw Error(error.message);
    return {history: Array.from(words.slice(0, history.size / 4)),
      receipt: Array.from(words.slice(history.size / 4))};
  } finally {
    if (scopeOpen) await device.popErrorScope();
    for (const buffer of owned) buffer.destroy();
  }
}
