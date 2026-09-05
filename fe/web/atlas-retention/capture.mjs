// Diagnostic transport only. Production retention never reads geometry back.
// Storage-only actor buffers need a copy pass before staging readback.
export async function captureRetention(surface) {
  await surface.live();
  await surface._render();
  const gpu = surface._gpu;
  if (gpu.passRecords.length !== 6) throw Error('expected six retention passes');
  const {device, resourceBuffers} = gpu;
  const names = ['header', 'directory', 'copied', 'points', 'triangles',
    'resident_points', 'resident_triangles'];
  const owned = [], data = {};
  device.pushErrorScope('validation');
  let scope = true;
  try {
    const module = device.createShaderModule({code: `
      @group(0) @binding(0) var<storage,read> src:array<u32>;
      @group(0) @binding(1) var<storage,read_write> dst:array<u32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
        if(id.x<arrayLength(&src)){dst[id.x]=src[id.x];}
      }`});
    const pipeline = await device.createComputePipelineAsync({layout:'auto', compute:{module, entryPoint:'main'}});
    for (const name of names) {
      const source = resourceBuffers.get(name);
      if (!source) throw Error(`missing Fe buffer ${name}`);
      const copy = device.createBuffer({size:source.size, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const staging = device.createBuffer({size:source.size, usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      owned.push(copy, staging);
      const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0), entries:[
        {binding:0, resource:{buffer:source}}, {binding:1, resource:{buffer:copy}},
      ]});
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(source.size / 256)); pass.end();
      encoder.copyBufferToBuffer(copy, 0, staging, 0, source.size);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      data[name] = [...new Uint32Array(staging.getMappedRange())]; staging.unmap();
    }
    const error = await device.popErrorScope(); scope = false;
    if (error) throw Error(error.message);
    if (surface._gpu !== gpu) throw Error('GPU generation changed during capture');
    return {manifest: surface._manifestUrl, passCount: gpu.passRecords.length,
      deviceStorageBindingLimit: device.limits.maxStorageBuffersPerShaderStage, ...data};
  } finally {
    for (const buffer of owned) buffer.destroy();
    if (scope) await device.popErrorScope();
  }
}
