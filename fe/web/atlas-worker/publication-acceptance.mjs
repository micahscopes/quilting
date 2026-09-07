// Independent integration test, not application loading or rendering policy.
// The emitted test fixture chooses the target and source range in Fe. We seed
// that fixture's Wasm source word and observe actual GPU pixel readback.
export async function verifyBufferPublicationSurface() {
  const runtime=await import('./generated-publication/fe-render-runtime.js');
  const surface=document.querySelector('fe-surface');
  if(!surface)throw Error('open the generated publication test site first');
  await surface._readyPromise;
  await surface.live();
  if(surface._mode!=='webgpu')throw Error('the acceptance requires actual WebGPU');
  const root=new URL('./generated-publication/',import.meta.url);
  const manifest=await (await fetch(new URL('manifest.json',root))).json();
  const {instance}=await WebAssembly.instantiateStreaming(fetch(new URL(manifest.artifacts.wasm,root)),{});
  const memory=instance.exports.memory;
  surface._bufferPublications=new runtime.BufferPublicationBindings(instance.exports,surface._resources);
  async function draw(tint) {
    const gpu=surface._gpu;
    surface._uniforms=[tint];
    const capture=await surface._presentOn(surface._liveContext,[tint],{width:2,height:2,format:gpu.format});
    try {
      await capture.buffer.mapAsync(GPUMapMode.READ);
      const pixels=Array.from(runtime.unpackCanvasReadback(
        new Uint8Array(capture.buffer.getMappedRange()),capture.width,capture.height,capture.bytesPerRow,capture.format));
      capture.buffer.unmap();
      return pixels;
    } finally {capture.buffer.destroy();}
  }
  new DataView(memory.buffer).setUint32(1024,0,true);
  const initial=await draw(1);
  new DataView(memory.buffer).setUint32(1024,7,true);
  const unchanged=await draw(1);
  const updated=await draw(2);
  const expected=value=>Array.from({length:4},()=>[value,0,255,255]).flat();
  if(JSON.stringify(initial)!==JSON.stringify(expected(0)))throw Error('initial GPU pixels');
  if(JSON.stringify(unchanged)!==JSON.stringify(initial))throw Error('unchanged revision was uploaded');
  if(JSON.stringify(updated)!==JSON.stringify(expected(7)))throw Error('Fe publication did not reach shader input');
  return {backend:surface._mode,initial,unchanged,updated,sourceWord:7,
    fixture:'compiler-generated raster with nominal buffer publication',
    scope:'controlled Wasm source bytes, not worker atlas generation'};
}
