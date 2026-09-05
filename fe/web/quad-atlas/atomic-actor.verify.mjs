// Diagnostic only: execute compiler output from FE_PRINT_ATOMIC_ACTOR.
// This is not a demo runtime, geometry provider, or authored shader fallback.
export async function verifyAtomicActor(device, fixture) {
  const names = fixture.passes.map(pass => pass.declaration.source_entry);
  if (names.join(',') !== 'initialize,claim,observe,paint') {
    throw new Error(`unexpected atomic actor: ${names}`);
  }
  const buffers = [];
  const resourceBuffers = new Map();
  device.pushErrorScope('validation');
  let scopeOpen = true;
  try {
    for (const resource of fixture.resources) {
      if (resource.artifact) throw new Error('the atomic fixture must initialize on GPU');
      const buffer = device.createBuffer({
        size: resource.length * resource.stride,
        // COPY_SRC is diagnostic capability only, not a production dependency.
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      buffers.push(buffer);
      resourceBuffers.set(resource.name, buffer);
    }
    if (resourceBuffers.size !== 2 || !resourceBuffers.has('claims') || !resourceBuffers.has('receipts')) {
      throw new Error('expected precisely the claims and receipts resources');
    }
    const prepared = [];
    for (const { declaration, wgsl } of fixture.passes.slice(0, 3)) {
      if (!declaration.dispatch || declaration.cycle || (declaration.repeat ?? 1) !== 1) {
        throw new Error('expected a single fixed dispatch per fixture stage');
      }
      const shader = device.createShaderModule({ code: wgsl });
      const info = await shader.getCompilationInfo();
      const errors = info.messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(errors.map(error => error.message).join('\n'));
      const pipeline = await device.createComputePipelineAsync({
        layout: 'auto', compute: { module: shader, entryPoint: declaration.layout.entry_point },
      });
      const entries = declaration.layout.bindings.map(binding => {
        if (binding.group !== 0) throw new Error('unexpected fixture bind group');
        let buffer;
        if (binding.role === 'resource') {
          buffer = resourceBuffers.get(binding.name);
          if (!buffer) throw new Error(`unresolved compiler resource ${binding.name}`);
        } else if (binding.role === 'output') {
          buffer = device.createBuffer({
            size: Math.max(4, binding.span), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          });
          buffers.push(buffer);
        } else {
          throw new Error('the fixture must not require host-authored state inputs');
        }
        return { binding: binding.binding, resource: { buffer } };
      });
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      prepared.push({ pipeline, bindGroup, dispatch: declaration.dispatch });
    }
    const byteCount = 16 + 64 * 4;
    const staging = device.createBuffer({ size: byteCount, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    buffers.push(staging);
    const observations = [];
    for (let generation = 0; generation < 5; ++generation) {
      const encoder = device.createCommandEncoder();
      for (const stage of prepared) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(stage.pipeline);
        pass.setBindGroup(0, stage.bindGroup);
        pass.dispatchWorkgroups(...stage.dispatch);
        pass.end();
      }
      encoder.copyBufferToBuffer(resourceBuffers.get('claims'), 0, staging, 0, 16);
      encoder.copyBufferToBuffer(resourceBuffers.get('receipts'), 0, staging, 16, 64 * 4);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const actual = [...new Uint32Array(staging.getMappedRange())];
      staging.unmap();
      const expected = [128, 0, 0, 0, ...Array(64).fill(128)];
      if (actual.some((value, index) => value !== expected[index])) {
        throw new Error(`atomic actor generation ${generation}: ${JSON.stringify(actual)}`);
      }
      observations.push({ generation, claims: actual.slice(0, 4), receipts: actual.slice(4) });
    }
    const error = await device.popErrorScope();
    scopeOpen = false;
    if (error) throw new Error(error.message);
    return { shaderBytes: fixture.passes.map(pass => new TextEncoder().encode(pass.wgsl).length), observations };
  } finally {
    for (const buffer of buffers) buffer.destroy();
    if (scopeOpen) await device.popErrorScope();
  }
}
