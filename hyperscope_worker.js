// Hyperscope Worker: loads WASM, handles spacetime slicing + Mobius transforms.

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

  if (type === 'get_shaders') {
    const vs = wasm.get_vertex_glsl();
    const fs_matcap = wasm.get_fragment_glsl('matcap');
    const fs_wire = wasm.get_fragment_glsl('wire');
    const fs_normals = wasm.get_fragment_glsl('normals');
    self.postMessage({ type: 'shaders', id, vs, fs_matcap, fs_wire, fs_normals });
    return;
  }

  if (type === 'set_sliver') {
    wasm.set_sliver_threshold(data.threshold);
    self.postMessage({ type: 'sliver_set', id });
    return;
  }

  if (type === 'build_atlas') {
    const { maxLodExp, mode } = data;
    const ms = wasm.build_atlas(maxLodExp, mode);
    self.postMessage({ type: 'atlas_built', id, ms });
    return;
  }

  if (type === 'create_hypermesh') {
    const result = wasm.create_hypermesh(data.name);
    self.postMessage({ type: 'hypermesh_created', id, result });
    return;
  }

  if (type === 'slice_and_transform') {
    const { normal, offset, transformType, params, overrideRes, vpMatrix, vpWidth, vpHeight, toroidal } = data;
    const result = wasm.slice_and_transform(
      new Float64Array(normal),
      offset,
      transformType,
      !!toroidal,
      new Float64Array(params),
      overrideRes,
      new Float64Array(vpMatrix || []),
      vpWidth || 0,
      vpHeight || 0,
    );
    self.postMessage({ type: 'batches', id, result });
    return;
  }

  if (type === 'load_gltf_data') {
    const result = wasm.load_gltf_data(new Uint8Array(data.bytes));
    self.postMessage({ type: 'gltf_loaded', id, result });
    return;
  }

  if (type === 'generate_single') {
    const { a, b, c } = data;
    wasm.generate_and_store_patch(a, b, c);
    self.postMessage({ type: 'generated', id });
    return;
  }
};
