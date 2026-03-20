// Web Worker: loads WASM, builds a portion of the atlas.
// Main thread sends a list of LOD triples to generate.
// Worker generates them and posts back results.

let wasm = null;

self.onmessage = async function(e) {
  const { type, data, id } = e.data;

  if (type === 'init') {
    const mod = await import('./pkg/quilting_wasm.js');
    await mod.default();
    wasm = mod;
    self.postMessage({ type: 'ready', id });
    return;
  }

  if (type === 'build_atlas') {
    // Build the full atlas for this worker's LOD range
    const { maxLodExp, mode } = data;
    const ms = wasm.build_atlas(maxLodExp, mode);
    self.postMessage({ type: 'atlas_built', id, ms });
    return;
  }

  if (type === 'generate_single') {
    const { a, b, c } = data;
    wasm.generate_and_store_patch(a, b, c);
    self.postMessage({ type: 'generated', id });
    return;
  }

  if (type === 'compute_batches') {
    const { positions, faces, transformType, params, overrideRes, vpMatrix, vpWidth, vpHeight } = data;
    const result = wasm.compute_mesh_batches(
      new Float64Array(positions),
      new Uint32Array(faces),
      transformType,
      new Float64Array(params),
      overrideRes,
      new Float64Array(vpMatrix || []),
      vpWidth || 0,
      vpHeight || 0,
    );
    self.postMessage({ type: 'batches', id, result });
    return;
  }
};
