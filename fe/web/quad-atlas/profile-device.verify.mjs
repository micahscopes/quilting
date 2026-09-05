// Diagnostic device only. Recompile exact served WGSL with the manifest's
// binding layouts; do not change shader source, dispatches, or geometry policy.
export async function createTimestampProfileDevice(surface) {
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter?.features.has('timestamp-query'))throw Error('timestamp-query unavailable');
  const device=await adapter.requestDevice({requiredFeatures:['timestamp-query']});
  device.pushErrorScope('validation');
  try {
    const passRecords=[];
    for(const record of surface._gpu.passRecords){
      const pass=record.pass;if(pass.layout.mode!=='compute')continue;
      const response=await fetch(record.shaderUrl);
      if(!response.ok)throw Error(`shader fetch failed: ${response.status}`);
      const code=await response.text();
      const module=device.createShaderModule({code});
      const group=device.createBindGroupLayout({entries:pass.layout.bindings.map(b=>({
        binding:b.binding,visibility:GPUShaderStage.COMPUTE,
        buffer:{type:b.access==='read'?'read-only-storage':'storage'},
      }))});
      const layout=device.createPipelineLayout({bindGroupLayouts:[group]});
      const pipeline=await device.createComputePipelineAsync({layout,
        compute:{module,entryPoint:pass.layout.entry_point}});
      passRecords.push({pass,pipeline});
    }
    const error=await device.popErrorScope();if(error)throw Error(error.message);
    return {device,passRecords};
  }catch(error){device.destroy();throw error;}
}
