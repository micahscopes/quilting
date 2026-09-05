// Diagnostic-only readback of actual Fe-generated buffers. This copy shader
// neither samples nor selects topology and is never part of the demo pipeline.
export async function captureCandidateAtlas(surface, params, triangular = false) {
  await surface.live();
  if (params) Object.assign(surface.params, params);
  await surface._render();
  const gpu = surface._gpu;
  if (gpu.passRecords.length !== 47) throw Error('expected the 47-pass indexed generator');
  const {device, resourceBuffers} = gpu;
  const manifest = await (await fetch(surface._manifestUrl)).json();
  const sampling = triangular ? 'sampling_workspace' : 'sampling';
  const names = ['receipt', 'points', 'triangles', 'repair_state', 'twins', sampling, 'candidate_bounds'];
  const owned = [], data = {};
  device.pushErrorScope('validation'); let scope = true;
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
      if (!source) throw Error(`missing generated buffer ${name}`);
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
    if (surface._gpu !== gpu) throw Error('generation changed during capture');
    const repair = data.repair_state, capacity = (data[sampling].length - 1) / 8;
    const count = triangular ? capacity : 2 * (2 ** (1 + Math.max(
      surface.params.bottom, surface.params.right, surface.params.top, surface.params.left))) ** 2;
    if (count > capacity) throw Error('truncated candidate job is not a valid acceptance case');
    return {
      manifest:surface._manifestUrl, passCount:gpu.passRecords.length, params:{...surface.params},
      globalResources:manifest.resources.length,
      maximumBindings:Math.max(...manifest.passes.map(p => p.layout.bindings.length)),
      deviceStorageBindingLimit:device.limits.maxStorageBuffersPerShaderStage,
      receipt:data.receipt, repair, twins:data.twins.slice(0, repair[1] * 3),
      points:Array.from({length:repair[0]}, (_,i) => data.points.slice(i*3, i*3+(triangular?3:2))),
      triangles:Array.from({length:repair[1]}, (_,i) => data.triangles.slice(i*4, i*4+3)),
      candidateIndex:{capacity, count, triangular, words:data.candidate_bounds,
        states:data[sampling].slice(capacity * (data[sampling][capacity * 8] === 0 ? 5 : 6),
          capacity * (data[sampling][capacity * 8] === 0 ? 5 : 6) + count),
        candidates:Array.from({length:count}, (_,i) => data[sampling].slice(i*5,i*5+5))},
    };
  } finally {
    for (const buffer of owned) buffer.destroy();
    if (scope) await device.popErrorScope();
  }
}
