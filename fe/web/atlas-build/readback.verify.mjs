// Bounded diagnostic observation, never imported by the application. Production
// buffers have storage permission but intentionally no COPY_SRC permission.
// This generic copy kernel reads them through that existing storage capability;
// it neither changes the atlas nor supplies geometry or scheduling decisions.
export async function readAtlasReceipts(device, resources) {
  const names = ['active', 'atlas_header', 'atlas_directory', 'tile_receipts', 'completion'];
  const owned = [];
  const make = (size, usage) => {
    const buffer = device.createBuffer({size, usage});
    owned.push(buffer);
    return buffer;
  };
  device.pushErrorScope('validation');
  let scopeOpen = true;
  try {
    const module = device.createShaderModule({code: `
      @group(0) @binding(0) var<storage, read> source: array<u32>;
      @group(0) @binding(1) var<storage, read_write> snapshot: array<u32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        if id.x < arrayLength(&source) && id.x < arrayLength(&snapshot) {
          snapshot[id.x] = source[id.x];
        }
      }
    `});
    const pipeline = await device.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint: 'main'}});
    const encoder = device.createCommandEncoder();
    const reads = [];
    let total = 0;
    for (const name of names) {
      const source = resources.get(name);
      if (!source || source.size % 4) throw Error(`invalid receipt buffer ${name}`);
      total += source.size;
      if (total > 1024 * 1024) throw Error('receipt observation exceeds 1 MiB budget');
      const snapshot = make(source.size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const mapped = make(source.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
        {binding: 0, resource: {buffer: source}}, {binding: 1, resource: {buffer: snapshot}},
      ]});
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(source.size / 4 / 64)); pass.end();
      encoder.copyBufferToBuffer(snapshot, 0, mapped, 0, source.size);
      reads.push({name, mapped});
    }
    device.queue.submit([encoder.finish()]);
    const result = {};
    for (const {name, mapped} of reads) {
      await mapped.mapAsync(GPUMapMode.READ);
      result[name] = Array.from(new Uint32Array(mapped.getMappedRange()));
      mapped.unmap();
    }
    const error = await device.popErrorScope(); scopeOpen = false;
    if (error) throw Error(error.message);
    return result;
  } finally {
    if (scopeOpen) await device.popErrorScope();
    for (const buffer of owned) buffer.destroy();
  }
}
