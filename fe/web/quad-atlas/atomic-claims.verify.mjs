// Diagnostic-only contention gate for the compiler's emitted atomic fixture.
// Not imported by the demo; no authored WGSL or geometry readback lives here.
// Supply the WGSL emitted by Sonatina's atomic_object_rmw_* regression and an
// existing WebGPU device. Never substitute a hand-written equivalent shader.
export async function verifyAtomicClaims(device, wgsl, entryPoint) {
  const receipts = [];
  const buffers = [];
  device.pushErrorScope('validation');
  let errorScopeOpen = true;
  try {
    const shader = device.createShaderModule({ code: wgsl });
    const info = await shader.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(message => message.message).join('\n'));
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: shader, entryPoint },
    });
    const claims = device.createBuffer({
      size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    buffers.push(claims);
    const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    buffers.push(staging);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: claims } }],
    });
    for (const workgroups of [1, 2, 16, 64, 255]) {
      for (const initialCount of [0, 0xfffffff0]) {
        // The high-bit minimum also detects an accidental signed atomicMin.
        device.queue.writeBuffer(claims, 0, new Uint32Array([initialCount, 0x80000000, 0x12345678, 0x9abcdef0]));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(claims, 0, staging, 0, 16);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const actual = [...new Uint32Array(staging.getMappedRange())];
        staging.unmap();
        const updates = 2 * 64 * workgroups; // Two preserved helper calls per invocation.
        const expected = [(initialCount + updates) >>> 0, 0, 0x12345678, 0x9abcdef0];
        if (actual.some((value, index) => value !== expected[index])) {
          throw new Error(`atomic contention failure: ${JSON.stringify({ workgroups, initialCount, expected, actual })}`);
        }
        receipts.push({ workgroups, invocations: 64 * workgroups, updates, initialCount, actual });
      }
    }
    const error = await device.popErrorScope();
    errorScopeOpen = false;
    if (error) throw new Error(error.message);
    return { shaderBytes: new TextEncoder().encode(wgsl).length, receipts };
  } finally {
    for (const buffer of buffers) buffer.destroy();
    if (errorScopeOpen) await device.popErrorScope();
  }
}
