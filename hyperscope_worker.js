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
    const fs_pbr = wasm.get_fragment_glsl('pbr');
    const fs_pick = wasm.get_fragment_glsl('pick');
    self.postMessage({ type: 'shaders', id, vs, fs_matcap, fs_wire, fs_normals, fs_pbr, fs_pick });
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

  if (type === 'build_atlas_subset') {
    const { maxLodExp, mode, workerIndex, numWorkers } = data;
    const ms = wasm.build_atlas_subset(maxLodExp, mode, workerIndex, numWorkers);
    self.postMessage({ type: 'atlas_subset_built', id, ms });
    return;
  }

  if (type === 'merge_atlas_bytes') {
    const ok = wasm.merge_atlas_bytes(new Uint8Array(data.bytes));
    self.postMessage({ type: 'atlas_merged', id, ok });
    return;
  }

  if (type === 'export_atlas_bytes') {
    const bytes = wasm.export_atlas_bytes();
    self.postMessage({ type: 'atlas_bytes', id, bytes: bytes.buffer }, [bytes.buffer]);
    return;
  }

  if (type === 'import_atlas_bytes') {
    const ok = wasm.import_atlas_bytes(new Uint8Array(data.bytes));
    self.postMessage({ type: 'atlas_imported', id, ok });
    return;
  }

  if (type === 'export_patches') {
    const patches = wasm.export_all_patches();
    self.postMessage({ type: 'patches_exported', id, patches });
    return;
  }

  if (type === 'extend_atlas') {
    const ms = wasm.extend_atlas(data.newLod);
    self.postMessage({ type: 'atlas_extended', id, ms });
    return;
  }

  if (type === 'create_hypermesh') {
    const result = wasm.create_hypermesh(data.name);
    self.postMessage({ type: 'hypermesh_created', id, result });
    return;
  }

  if (type === 'slice_and_transform') {
    const { normal, offset, transformType, params, overrideRes, vpMatrix, vpWidth, vpHeight, toroidal, tessDensity, screenAtten, minPxSub } = data;
    if (tessDensity != null) {
      wasm.set_tess_params(tessDensity, !!screenAtten);
      if (minPxSub != null) wasm.set_min_px_per_sub(minPxSub);
    }
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
    // Transfer flat buffers zero-copy to main thread
    const transferList = [];
    if (result.all_orig) transferList.push(result.all_orig.buffer);
    if (result.all_xform) transferList.push(result.all_xform.buffer);
    if (result.batch_meta) transferList.push(result.batch_meta.buffer);
    if (result.face_indices) transferList.push(result.face_indices.buffer);
    self.postMessage({ type: 'batches', id, result }, transferList);
    return;
  }

  if (type === 'load_gltf_data') {
    const result = wasm.load_gltf_data(new Uint8Array(data.bytes));
    self.postMessage({ type: 'gltf_loaded', id, result });
    return;
  }

  if (type === 'set_face_materials') {
    wasm.set_face_materials(new Int32Array(data.materials));
    self.postMessage({ type: 'face_materials_set', id });
    return;
  }

  if (type === 'generate_single') {
    const { a, b, c } = data;
    wasm.generate_and_store_patch(a, b, c);
    self.postMessage({ type: 'generated', id });
    return;
  }
};
